 clear; clc;
%% Initialization
feedPattern = [[-1 -1 -1];[1 -1 1];[-1 1 1];[1 1 -1]];
allPattern = [[-1 -1 -1];[1 -1 1];[-1 1 1];[1 1 -1];[-1 -1 1];[-1 1 -1];[1 -1 -1];[1 1 1]];

learningRate = 0.005;
M = 8; %1,2,4,8

V = zeros(1,3);
h = zeros(1,M);
W = randn(M,size(V,2));

theta1 = zeros(1,3);
theta2 = zeros(1,M);

nEpoch = 100;
miniBatches = 20;
dklCounter = 5;
k = 200;
dklArray = [];

validationIteration = 3000;
validationK = 2000;

for counter = 1:dklCounter
    %% Training
    for epoch = 1:nEpoch

        wDiff = zeros(M,size(V,2));
        theta1Diff = zeros(1,3);
        theta2Diff = zeros(1,M);

        for iteration = 1:miniBatches
        
            % Calculate V0 and h0

            selectedPattern = fix(rand()*size(feedPattern,1))+1;
            V0 = feedPattern(selectedPattern,:);
            h0 = zeros(1,M);
            V = V0;

            for i = 1:size(h,2)
                bh = W(i,:)*V0'-theta2(i);
                p = (1/(1+exp(-2*bh)));
                if(rand < p)
                    h0(i) = 1;
                else
                    h0(i) =-1;
                end
            end

            h = h0;

            % CDK
            for t = 1:k
            
                % Update visible neurons
                for i = 1:size(V,2)
                    bv = h*W(:,i) - theta1(i);
                    p = (1/(1+exp(-2*bv)));
                    if(rand < p)
                        V(i) = +1;
                    else
                        V(i) = -1;
                    end
                end

                % Update hidden neurons
                for i = 1:size(h,2)
                    bh = W(i,:)*V'-theta2(i);
                    p = (1/(1+exp(-2*bh)));
                    if(rand < p)
                        h(i) = 1;
                    else
                        h(i) = -1;
                    end
                end
            
            end
        
            % Calculate weight and threshold difference
        
            bh0 = (V0*W'-theta2)';
            bhk = (V*W' -theta2)';

            wDiff = wDiff + learningRate*((tanh(bh0)*V0)-(tanh(bhk)*V));
            theta1Diff = theta1Diff -learningRate*(V0-V);
            theta2Diff = theta2Diff -learningRate*(tanh(bh0)-tanh(bhk))';
        
        
        
        end

        % Update weights and threshold
        W = W + wDiff;
        theta1 = theta1 + theta1Diff;
        theta2 = theta2 + theta2Diff;

    end


    %% Validation
    pTest = zeros(1,size(allPattern,1));
    for iteration = 1:validationIteration

        V = zeros(1,3);
        h = zeros(1,M);
        
        % Calculate V0 and h0
        selectedPattern = fix(rand()*size(allPattern,1))+1;
        V0 = allPattern(selectedPattern,:);
        h0 = zeros(1,M);

        for i = 1:size(h,2)
            bh = W(i,:)*V0'-theta2(i);
            p = (1/(1+exp(-2*bh)));
            if(rand < p)
                h0(i) = 1;
            else
                h0(i) =-1;
            end
        end

        h = h0;

        % CDK
        for t = 1:validationK
            
            % Update visible neurons
            for i = 1:size(V,2)
                bv = h*W(:,i) - theta1(i);
                p = (1/(1+exp(-2*bv)));
                if(rand < p)
                    V(i) = +1;
                else
                    V(i) = -1;
                end
            end

            % Update hidden neurons
            for i = 1:size(h,2)
                bh = W(i,:)*V'-theta2(i);
                p = (1/(1+exp(-2*bh)));
                if(rand < p)
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end

            convergedPattern = find(ismember(allPattern,V,'rows'));
            pTest(convergedPattern) = pTest(convergedPattern) + 1;
            
        end
    end

    %%Dkl
    dkl = 0;
    Pdata = 0.25;
    for i = 1:4
        dkl = dkl + (Pdata*log(Pdata/(pTest(i)/sum(pTest))));
    end

    dklArray = [dklArray;dkl];

    %% Summary
    sprintf('M : %d, DKL iteration : %i DKL : %0.5f',M,counter,dkl)
    
end

%% Summary
disp("---- Parameter ----")
sprintf('M : %d, Learning rate : %0.2f, Epoch : %d, batches size: %d, k : %d',M,learningRate,nEpoch,miniBatches,k)
sprintf('AVG DKL : %0.5f',counter,mean(dklArray))
disp("-------------------")