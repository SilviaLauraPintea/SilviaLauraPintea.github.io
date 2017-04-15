require 'optim'
require 'math'
require 'unsup'
class = require 'class'	

-- TODO: For now just scalar lengthscales implemented in Torch.
GaussianProcess = class('GaussianProcess')

function GaussianProcess:__init(data, labels, train, clusters, initNoise, initScale, algo)
--	torch.manualSeed(0)
	self.algo_ = algo
	self.symmetric = false
	self.whiten = true

	self.mu_ = 0 -- Regularizer for the gradient.
	self.miniBatchSize_ = 1
	self.clusters_ = clusters
	self.train_ = train
	self.mean_ = torch.Tensor(data:size(2)):zero()
	self.std_ = torch.Tensor(data:size(2)):zero()
	self.labelsVariance_ = labels:var()

	-- If train the cluster the data.
	if(self.train_) then
		self.xvalData_ = data
		self.xvalLabels_ = labels
		self:clusterData(self.xvalData_, self.xvalLabels_)

		self.data_ = self:whitenData(self.data_, true)
		self.xvalData_ = self:whitenData(self.xvalData_, false)
	else
		self.labels_ = labels
		self.data_ = self:whitenData(data, false)
	end	

	self.alpha_ = torch.Tensor(self.clusters_):zero()
	self.precision_ = torch.Tensor(self.clusters_):zero()	
	local xvalscale = initScale
	self.noise_ = initNoise

	for r = 1, self.precision_:size(1) do
		self.precision_[r] = xvalscale; 
	end
	print("[Gaussian Process] Initial precision:", self.precision_);
end
	
function GaussianProcess:whitenData(data, getStats)
	if(self.whiten) then
		print("mean & std: ", self.mean_, self.std_)

		if(getStats) then
			self.mean_ = data:mean(1)
			self.std_ = data:std(1) 

			if(data:size(1) == 1) then
				self.std_ = data 

				-- Do not whiten if only 1 sample
				return data
			end
		end
	
		for r = 1, data:size(1) do
			data[r] = data[r]:csub(self.mean_):cdiv(self.std_)
		end
	end
	return data
end

function GaussianProcess:inverse(kernel)
	local inverse = {}
	if (self.symmetric) then
		kernel = (kernel + kernel:transpose(1,2)) * 0.5
		local chol = torch.potrf(kernel) -- Cholesky decomposition
		inverse = torch.potri(chol) 
	else
		inverse = torch.inverse(kernel)
	end 
	return inverse;
end

function GaussianProcess:train()
	local kernel = self:trainKernel()
	local kernelInv = self:inverse(kernel)
	self.alpha_ = kernelInv * self.labels_;
end

function GaussianProcess:sqexp(sample1, sample2, lengthscale)
	if(lengthscale == 0) then
		lengthscale = 1.0e-7 -- numeric problem.
	end
	local diff = (sample1 - sample2)
	local squared = diff:dot(diff)
	local similarity = math.exp(-1.0/(2.0 * lengthscale * lengthscale) * squared);

	assert(similarity == similarity and similarity ~= nil)
	return similarity	
end

function GaussianProcess:optimizeCovar()
	parameters =
	{
		learningRate = 1e-2;
		momentum = 0.9
	}		

	self.miniBatchSize_ = math.min(self.miniBatchSize_, self.xvalData_:size(1))
	local precision = self.precision_ 
	local miniBatchData = self.xvalData_:split(self.miniBatchSize_)
	local miniBatchLabels = self.xvalLabels_:split(self.miniBatchSize_)

	for i,v in ipairs(miniBatchData) do

		function optimFunction(precision) 
			self.precision_ = precision
		
			-- Define the loss function
			f = function (precision)
				self:train()
				prediction = self:predict(v)

				return self:loss(prediction, miniBatchLabels[i])
			end

			-- Define th gradient wrt precision
			df = function (precision)
				return self:estimateGrad(precision, v, miniBatchLabels[i]) 
			end
			
			return f(precision), df(precision)
		end
	
		if (self.algo_ == 'sgd') then
			self.precision_ = optim.sgd(optimFunction, precision, parameters)					
		elseif (self.algo_ == 'cg') then
			self.precision_ = optim.cg(optimFunction, precision, parameters)					
		elseif (self.algo_ == 'lbfgs') then
			self.precision_ = optim.lbfgs(optimFunction, precision, parameters)					
		end

	end -- over the mini-batches
	print("Found precision:", self.precision_);
end

function GaussianProcess:estimateGrad(precision, miniBatchData, miniBatchLabels)
	local kernel = self:trainKernel()
	local kernelInverse = self:inverse(kernel)
	local kstar = self:testKernel(miniBatchData)
	local alpha = self.alpha_
	local prediction = self:predict(miniBatchData) -- [M x N] x [N x 1] 
	local grad = torch.Tensor(precision:size()):zero()
		
	for i = 1, self.data_:size(1) do
		if(precision[i] == 0) then 
			precision[i] = 1.0e-7
		end

		local xi = self.data_[i]  

		local gradi = 0
		for n = 1, miniBatchData:size(1) do
			-- kstar derivative
			local xn = miniBatchData[n]  
			local diffNI = xn - xi
			local dkstar = 1.0 / (precision[i] * precision[i] * precision[i]) * diffNI:dot(diffNI) * kstar[i][n] 
			-- kstar derivative

			-- alpha derivative
			local dKinv = torch.Tensor(kernelInverse:size(1), kernelInverse:size(2)):zero()	
			for m = 1, self.data_:size(1) do
				for j = 1, self.data_:size(1) do
					local xj = self.data_[j]
					for k = 1, self.data_:size(1) do
						local xk = self.data_[k]
						local dKkj = 0
							if(i == k) then
								local diffIJ = xk - xj
								dKkj = 1.0 /(precision[k] * precision[k] * precision[k])  * diffIJ:dot(diffIJ)  * kernel[k][j]
							end		
						dKinv[m][j] = dKinv[m][j] + (-1.0 * kernelInverse[m][k] * dKkj)
					end
				end
			end
			
			local findKinv = torch.Tensor(kernelInverse:size(1), kernelInverse:size(2)):zero()	
			for m = 1, self.data_:size(1) do
				for j = 1, self.data_:size(1) do
					for k = 1, self.data_:size(1) do
						findKinv[m][j] = findKinv[m][j] + dKinv[m][k] * kernelInverse[k][j]
					end
				end
			end
		
			local dalphakstar = 0
			local dalpha = torch.Tensor(kernelInverse:size(1), 1):zero()	
			for m = 1, self.data_:size(1) do
				for j = 1, self.data_:size(1) do
					dalpha[m][1] = dalpha[m][1] + (findKinv[m][j] * self.labels_[j][1]) 
				end
				dalphakstar = dalphakstar + dalpha[m][1] * kstar[m][n]
			end
			--- alpha derivative

			-- loss derivative
			local dloss = 2.0 * (prediction[n][1] - miniBatchLabels[n][1])

			-- final gradient
			gradi = gradi + dloss * (dkstar * alpha[i][1] + dalphakstar)
			grad[i] = grad[i] + gradi;
		end -- loop over samples in mini-batch

		-- Add the regularizer.
		grad[i] = grad[i] + 2.0 * self.mu_ * precision[i]; 
	end -- loop over clusters

	assert(grad:norm() == grad:norm() and grad:norm()~=nil)
	return grad
end

function GaussianProcess:nrmse(prediction, labels)
	local loss = 0;
	local diff = prediction - labels
	loss = diff:dot(diff)
	if self.labelsVariance_ ~= 0 then
		loss = loss / self.labelsVariance_
	end
	loss = loss / prediction:size(1) 
	loss = math.sqrt(loss)
	return loss
end

function GaussianProcess:loss(prediction, labels)
	local loss = 0;
	local diff = prediction - labels
	loss = diff:dot(diff)
	assert(loss == loss and loss ~= nil)
	return loss
end

 
function GaussianProcess:predict(miniBatchData)
	local prediction = torch.Tensor(miniBatchData:size(1), 1):zero()
	local kstar = self:testKernel(miniBatchData)
	local prediction = kstar:transpose(1,2) * self.alpha_
	return prediction
end

function GaussianProcess:trainKernel()
	local kernel = torch.Tensor(self.data_:size(1), self.data_:size(1)):zero()
	for i = 1, kernel:size(1)  do
		for j = 1, kernel:size(2)  do
			kernel[i][j] = self:sqexp(self.data_[i], self.data_[j], self.precision_[i])
		end -- over clusters
	end -- over clusters
	
	local kernelDiag = torch.eye(self.data_:size(1))
	kernelDiag = kernelDiag * self.noise_;
	kernel = kernel + kernelDiag;
	return kernel
end

function GaussianProcess:testKernel(miniBatchData)
	local kernel = torch.Tensor(self.data_:size(1), miniBatchData:size(1)):zero()
	for i = 1, kernel:size(1)  do
		for n = 1, kernel:size(2)  do
			kernel[i][n] = self:sqexp(self.data_[i], miniBatchData[n], self.precision_[i])
		end -- over mini-batch data
	end -- over clusters

	return kernel
end

function GaussianProcess:clusterData(data, labels)	
	local tmpdata = unsup.kmeans(data, self.clusters_, 100);
	local nonunique = {}
	for i = 1, tmpdata:size(1) do
		for j = i+1, tmpdata:size(1) do
			local norm = (tmpdata[i] - tmpdata[j]):norm()		
			if(norm == 0) then
				nonunique[j] = 1 
			end
		end
	end
	local uniqueSize = 0
	for i = 1, tmpdata:size(1) do
		if(nonunique[i] == nil) then
			uniqueSize = uniqueSize + 1
		end
	end
	
	self.clusters_ = uniqueSize
	self.data_ = torch.Tensor(uniqueSize, tmpdata:size(2))
	local index = 1
	for i = 1, tmpdata:size(1) do
		if(nonunique[i] == nil) then
			self.data_[index] = tmpdata[i]
			index = index + 1 
		end
	end

	self.labels_ = torch.Tensor(self.clusters_, 1):zero()
	for i = 1, self.data_:size(1) do
		local mindist = 1.0e+7
		for j = 1, data:size(1) do
			local dist = torch.dist(self.data_[i], data[j])
			if dist < mindist then
				mindist = dist
				self.labels_[i] = labels[j]
			end
		end
	end 
end



