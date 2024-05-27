%% PROJECT 4 : Learning data quality in EEG
% Filippo Canderle, Cecilia Rossi

clc; close all; clear all;

%% Files Description:
% Each hdf5 file contains 3 dataset: 
%  - 'ID' : with patients IDs (which could be repeated) : 1x200
%
%  - 'dataset' : with the EEG data, so with a dimention of 
%                 23(TUAR)/21(TUAB) (channels)
%                 x
%                 500 (samples: and since Fs = 250 --> 2 seconds windows)
%                 x
%                 200 (patients)
% 
%  - 'label' : that contains the labels for each patient : 1x200

%% IMPORT TUAR
% -----
% labels legend:
% 	label 0 = artifact
% 	label 1 = clean
% Channel set: A1, A2, C3, C4, CZ, F3, F4, F7, F8, FP1, FP2, FZ, O1, O2, P3, P4, PZ, T1, T2, T3, T4, T5, T6
% Sample frequency: 250 Hz

data_tuar = h5read('Data/TUAR.hdf5', '/dataset');
label_tuar = h5read('Data/TUAR.hdf5', '/label');
id_tuar = h5read('Data/TUAR.hdf5', '/ID')';
id_tuar_new = nan(1,size(id_tuar,2));
for el=1:size(id_tuar,2)
    id_tuar_new(el) = str2double(id_tuar{1,el});
end %for
id_tuar = id_tuar_new;
clear id_tuar_new;

uniq_id_tuar = unique(id_tuar); 
%esce un vettore di 27 elem...quindi i
%pazienti in realtà sono 27, ma ci sono osservazioni ripetute??

%% IMPORT TUAB
% ----
% labels legend:
% 	label 0 = abnormal
% 	label 1 = normal
% Channel set: A1, A2, C3, C4, CZ, F3, F4, F7, F8, FP1, FP2, FZ, O1, O2, P3, P4, PZ, T3, T4, T5, T6  
% Sample frequency: 250 Hz

%TUAB ha 2 canali in meno (T1 e T2)
data_tuab = h5read('Data/TUAB.hdf5', '/dataset'); 
label_tuab = h5read('Data/TUAB.hdf5', '/label'); 
id_tuab = h5read('Data/TUAB.hdf5', '/ID')';
id_tuab_new = nan(1,size(id_tuab,2));
for el=1:size(id_tuab,2)
    id_tuab_new(el) = str2double(id_tuab{1,el});
end %
id_tuab = id_tuab_new;
clear id_tuab_new;

uniq_id_tuab = unique(id_tuab);

%% Sampling Frequency
Fs = 250; %[Hz], so the signal is recorded in 2 seconds windows
nchan_tuar = size(data_tuar,1);
nchan_tuab = size(data_tuab,1);
T = size(data_tuar,2);
npaz = size(id_tuar,2); %the two datasets contain the same number of patients, even if the patients are different

%% Comparing the Results of the Two Pipelines
score_tabs = cell(1,2);

%% DATA FUSION

%Using this code, we realized that the patients IDs in the two datasets are
%different, so that there's no correspondence between the patients of the
%two datasets

% for y = 1:length(id_tuab)
%     paziente = id_tuab(y);
%     uguali = find(id_tuar == paziente);
%     if length(uguali) == 0
%         disp(['Paziente: ', num2str(paziente), ' corrispondeze: ', num2str(length(uguali))]);
%     end %if
% end %for

%% DATA FUSION
%%Assuming that all the Clean signals are also Normal (so not pathological)
%and viceversa, we create new classes: 

%  - Clean & Normal : class 1
%  - Artifact : class 2
%  - Abnormal : class 3

%Changing the labels
label_tuab(label_tuab == 0) = 3; %placing abnormal = 3
label_tuar(label_tuar == 0) = 2; % artefact = 2
%Concateno tutte le matrici
%ignoro i canali T1 e T2, visto che li ha solo Tuar e non Tuab --> ind 18 e
%19
data = nan(nchan_tuab, T, 2*npaz);
data(:, :, 1:npaz) = data_tuab;
data(1:17, :, npaz+1:2*npaz) = data_tuar(1:17,:,:); 
data(18:21, :, npaz+1:2*npaz) = data_tuar(20:23,:,:);

label = nan(2*npaz,1); 
label(1:npaz) = label_tuab; 
label(npaz+1:2*npaz) = label_tuar;

id = nan(1, 2*npaz);
id(1:npaz) = id_tuab;
id(npaz+1:2*npaz) = id_tuar;

%In order to make the classification more unbiased, we can apply a
%permutation of the 400 

new_data = nan(2*npaz*nchan_tuab, T);
new_label = nan(2*npaz*nchan_tuab,1);
for paz = 1:2*npaz
    new_data((paz-1)*nchan_tuab + 1: paz*nchan_tuab, :) = data(:,:,paz);
    new_label((paz-1)*nchan_tuab + 1: paz*nchan_tuab, :) = label(paz)*ones(1,nchan_tuab);
end %for

data = new_data;
label = new_label;
clear new_data new_label;

permutation = randperm(2*npaz*nchan_tuab);
data = data(permutation, :);
label = label(permutation);

%% See if the dataset is well balanced
disp('Creating the dataset...');
num_class1 = length(squeeze(label(label == 1)));
num_class2 = length(squeeze(label(label == 2)));
num_class3 = length(squeeze(label(label == 3)));
disp(['Number of samples of the first class (Clean & Normal): ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artifact): ', num2str(num_class2)]);
disp(['Number of samples of the third class (Abnormal): ', num2str(num_class3)]);
 
%% Splitting between train and test

per_train = 0.75;
train_dim = 0.75*length(label);

training_data = data(1:train_dim,:);
training_label = label(1:train_dim);
test_data = data(train_dim+1:length(label),:);
test_label = label(train_dim+1:length(label));

disp('Splitting the dataset between Train and Test set...');
num_class1 = length(squeeze(training_label(training_label == 1)));
num_class2 = length(squeeze(training_label(training_label == 2)));
num_class3 = length(squeeze(training_label(training_label == 3)));
disp(['Number of samples of the first class (Clean & Normal) in the Training Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artifact) in the Training Set: ', num2str(num_class2)]);
disp(['Number of samples of the third class (Abnormal) in the Training Set: ', num2str(num_class3)]);

num_class1 = length(squeeze(test_label(test_label == 1)));
num_class2 = length(squeeze(test_label(test_label == 2)));
num_class3 = length(squeeze(test_label(test_label == 3)));
disp(['Number of samples of the first class (Clean & Normal) in the Test Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artifact) in the Test Set: ', num2str(num_class2)]);
disp(['Number of samples of the third class (Abnormal) in the Test Set: ', num2str(num_class3)]);

%There is a certain balance. 

%% PIPELINE 1
disp("PIPELINE 1...");
%% First Classification: (Clean & normal) VS (artifact & abnormal)

%%Creation of Labels for the First Classification
%Bisogna creare delle label apposite -> 1 solo per Clean & normal e --> 0
%per gli altri
label_training_1_pip1 = training_label; 
label_training_1_pip1(label_training_1_pip1 ~= 1 ) = 0;

disp('First Classification between Clean&Normal and Artifact & Abnormal');
num_class1 = length(squeeze(label_training_1_pip1(label_training_1_pip1 == 1)));
num_class2 = length(squeeze(label_training_1_pip1(label_training_1_pip1 == 0)));
disp(['Number of samples of the first class (Clean & Normal) in the Training Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artifact & Abnormal) in the Training Set: ', num2str(num_class2)]);
num_class1 = length(squeeze(test_label(test_label == 1)));
num_class2 = length(squeeze(test_label(test_label ~= 1)));
disp(['Number of samples of the first class (Clean & Normal) in the Test Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artifact & Abnormal) in the Test Set: ', num2str(num_class2)]);


%%Training the Model
SVMmodel1_pip1 = fitcsvm(training_data, label_training_1_pip1);

[predicted_labels_train_1_pip1, score_train_1_pip1, cost_train_1_pip1] = predict(SVMmodel1_pip1, training_data);
%[predicted_labels_test_1, score_test_1, cost_test_1] = predict(SVMmodel1, test_data);
disp('End First Classification');

%% Second Classification: Artefact VS Abnormal
%We need to select only the patients with predicted label 0

paz_not_clean_and_correct_train = find(predicted_labels_train_1_pip1 == 0); %ritorna gli indici
paz_not_clean_and_correct_test = find(test_label ~= 1); %ritorna gli indici
%%non usare i valori predetti, ma il gold standard

%Creating a new training (considering the predicted label by the first classification)
%and test set (by using the gold standars)
label_training_2_pip1 = training_label(paz_not_clean_and_correct_train);
label_test_2_pip1 = test_label(paz_not_clean_and_correct_test);
%E' giusto fare così o semplicemente poniamo tutti gli altri (not artefact)
% == 1

training_data_2_pip1 = training_data(paz_not_clean_and_correct_train, :);
test_data_2_pip1 = test_data(paz_not_clean_and_correct_test, :);

%changing the labels
label_training_2_pip1(label_training_2_pip1 == 2) = 1; %ARTEFACT
label_training_2_pip1(label_training_2_pip1 == 3) = 0; %ABNORMAL
label_test_2_pip1(label_test_2_pip1 == 2) = 1;
label_test_2_pip1(label_test_2_pip1 == 3) = 0; 

disp('Second Classification between Artefact and Abnormal');
num_class1 = length(squeeze(label_training_2_pip1(label_training_2_pip1 == 1)));
num_class2 = length(squeeze(label_training_2_pip1(label_training_2_pip1 == 0)));
disp(['Number of samples of the first class (Artefact) in the Training Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Abnormal) in the Training Set: ', num2str(num_class2)]);
num_class1 = length(squeeze(label_test_2_pip1(label_test_2_pip1 == 1)));
num_class2 = length(squeeze(label_test_2_pip1(label_test_2_pip1 == 0)));
disp(['Number of samples of the first class (Artefact) in the Test Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Abnormal) in the Test Set: ', num2str(num_class2)]);

%%Training the Model
SVMmodel2_pip1 = fitcsvm(training_data_2_pip1, label_training_2_pip1);
[predicted_labels_train_2_pip1, score_train_2_pip1, cost_train_2_pip1] = predict(SVMmodel2_pip1, training_data_2_pip1);
[predicted_labels_test_2_pip1, score_test_2_pip1, cost_test_2_pip1] = predict(SVMmodel2_pip1, test_data_2_pip1);
disp('End Second Classification');

%% Performance
[TP_train_pip1, TN_train_pip1, FN_train_pip1, FP_train_pip1, error_train_pip1] = modelResults(label_training_2_pip1, predicted_labels_train_2_pip1);
[TP_test_pip1, TN_test_pip1, FN_test_pip1, FP_test_pip1, error_test_pip1] = modelResults(label_test_2_pip1, predicted_labels_test_2_pip1);
score_tabs = resultStruct(score_tabs, 1, error_train_pip1, error_test_pip1, TP_train_pip1, TN_train_pip1, FN_train_pip1, FP_train_pip1, TP_test_pip1, TN_test_pip1, FP_test_pip1, FN_test_pip1);

%% PIPELINE 2: 
disp("PIPELINE 2...");
%% First Classification: ((Clean & Normal) & Abnormal) VS Artefact

%%Creation of Labels for the First Classification
% 1: for Clean and Normal & Abnormal
% 0: Artefact
label_training_1_pip2 = training_label; 
label_test_1_pip2 = test_label;
label_training_1_pip2(label_training_1_pip2 ~= 2 ) = 1;
label_training_1_pip2(label_training_1_pip2 == 2) = 0;

disp('First Classification between (Clean & Normal & Abnormal) and Artefact');
num_class1 = length(squeeze(label_training_1_pip2(label_training_1_pip2 == 1)));
num_class2 = length(squeeze(label_training_1_pip2(label_training_1_pip2 == 0)));
disp(['Number of samples of the first class (Clean & Normal & Abnormal) in the Training Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Artefact) in the Training Set: ', num2str(num_class2)]);

%%Training the Model
SVMmodel1_pip2 = fitcsvm(training_data, label_training_1_pip2);
[predicted_labels_train_1_pip2, score_train_1_pip2, cost_train_1_pip2] = predict(SVMmodel1_pip2, training_data);
%[predicted_labels_test_1_pip2, score_test_1_pip2, cost_test_1_pip2] = predict(SVMmodel1_pip2, test_data);
disp('End Fist Classification');

%% Second Classification: (Clean & Normal) VS Abnormal
%We need to select only the patients with predicted label 0

paz_not_artifact_train_pip2 = find(predicted_labels_train_1_pip2 == 1); %ritorna gli indici
paz_not_artifact_test_pip2 = find(test_label ~= 2); %ritorna gli indici

%Creating a new training and test set
label_training_2_pip2 = training_label(paz_not_artifact_train_pip2);
label_test_2_pip2 = test_label(paz_not_artifact_test_pip2);

training_data_2_pip2 = training_data(paz_not_artifact_train_pip2, :);
test_data_2_pip2 = test_data(paz_not_artifact_test_pip2, :);

%changing the labels
label_training_2_pip2(label_training_2_pip2 ~= 3 ) = 1; %we have to do it because there could be some sample of class 2, that were incorrectly classified in the first step
label_training_2_pip2(label_training_2_pip2 == 3) = 0; %ABNORMAL
label_test_2_pip2(label_test_2_pip2 == 3) = 0; 

disp('Second Classification between (Clean & Normal) and Abnormal');
num_class1 = length(squeeze(label_training_2_pip2(label_training_2_pip2 == 1)));
num_class2 = length(squeeze(label_training_2_pip2(label_training_2_pip2 == 0)));
disp(['Number of samples of the first class (Clean & Normal) in the Training Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Abnormal) in the Training Set: ', num2str(num_class2)]);
num_class1 = length(squeeze(label_test_2_pip2(label_test_2_pip2 == 1)));
num_class2 = length(squeeze(label_test_2_pip2(label_test_2_pip2 == 0)));
disp(['Number of samples of the first class (Clean & Normal) in the Test Set: ', num2str(num_class1)]);
disp(['Number of samples of the second class (Abnormal) in the Test Set: ', num2str(num_class2)]);

%%Training the Model
SVMmodel2_pip2 = fitcsvm(training_data_2_pip2, label_training_2_pip2);
[predicted_labels_train_2_pip2, score_train_2_pip2, cost_train_2_pip2] = predict(SVMmodel2_pip2, training_data_2_pip2);
[predicted_labels_test_2_pip2, score_test_2_pip2, cost_test_2_pip2] = predict(SVMmodel2_pip2, test_data_2_pip2);
disp('End Second Classification');

%% Results of the Second Pipeline
[TP_train_2_pip2, TN_train_2_pip2, FN_train_2_pip2, FP_train_2_pip2, error_train_2_pip2] = modelResults(label_training_2_pip2, predicted_labels_train_2_pip2);
[TP_test_2_pip2, TN_test_2_pip2, FN_test_2_pip2, FP_test_2_pip2, error_test_2_pip2] = modelResults(label_test_2_pip2, predicted_labels_test_2_pip2);
score_tabs = resultStruct(score_tabs, 2, error_train_2_pip2, error_test_2_pip2, TP_train_2_pip2, TN_train_2_pip2, FN_train_2_pip2, FP_train_2_pip2, TP_test_2_pip2, TN_test_2_pip2, FP_test_2_pip2, FN_test_2_pip2);


%% Function modelResults
function [TP, TN, FN, FP, error] = modelResults(y, predicted_labels)
% created in order to compute automatically all the result parameters
% as TP, TN, FN, FP also if using and comparing more classificators

error = 0;
TP = 0; TN = 0; FN =0 ; FP = 0; 
        
for i = 1:length(predicted_labels)
    if predicted_labels(i) ~= y(i)
        error = error + 1;
        if predicted_labels(i) == 1
            FN = FN + 1;
        else
            FP = FP + 1;
        end %if/else
    else
        if predicted_labels(i) == 1
            TN = TN + 1;
        else
            TP = TP + 1;
        end %if/else
    end %if/else (outside)
end %for
error = error / length(y);

end %function 

%% Function resultStruct
function score_tabs = resultStruct(score_tabs, run, train_error, test_error, TP_train, TN_train, FN_train, FP_train, TP_test, TN_test, FP_test, FN_test)

%if run = 1: ho che 1 = Artefact, 0 = Abnormal
%if run = 2: ho che 1 = Clean and Normal, 0 = Abnormal

%On Train
score_tabs{1, run}.TRAIN_ERROR = train_error;
score_tabs{1, run}.TP_train = TP_train;
score_tabs{1, run}.TN_train = TN_train;
score_tabs{1, run}.FP_train = FP_train;
score_tabs{1, run}.FN_train = FN_train; 
score_tabs{1, run}.Accuracy_Train = ((TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train));

%if 
% On Test
score_tabs{1, run}.TEST_ERROR = test_error;
score_tabs{1, run}.TP_test = TP_test;
score_tabs{1, run}.TN_test = TN_test;
score_tabs{1, run}.FP_test = FP_test;
score_tabs{1, run}.FN_test = FN_test; 
score_tabs{1, run}.Accuracy_Test = ((TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test));


end %function
