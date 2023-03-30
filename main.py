import libemg
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Let's load in a single subject's worth of data to illustrate the process
    # of finding the best feature parameter

    dataset = libemg.datasets._3DCDataset(redownload=False)
    odh = dataset.prepare_data(subjects_values = ["1"])


    # For this process, lets split the dataset into training and testing sets using the 
    # "sets" metadata tag.
    training_odh = odh.isolate_data("sets", [0])
    test_odh = odh.isolate_data("sets", [1])

    # For the feature parameter optimization, let further split the training dataset into a 
    # "train" and "validation" set using the first three reps for training and the final rep
    # for validation. 
    valid_odh  = training_odh.isolate_data("reps", [3])
    train_odh  = training_odh.isolate_data("reps", [0,1,2])
    # Given that these are stored data from as Thalmic labs myoarmband, lets use a window size of
    # 50 samples and an increment of 10 samples.
    valid_windows, valid_metadata = valid_odh.parse_windows(50, 10)
    train_windows, train_metadata = train_odh.parse_windows(50, 10)


    # Let's choose to optimize for the Willison's amplitude (WAMP) feature threshold. This parameter
    # signifies the threshold that the absolute value of the derivative must exceed to count
    # as a fast changing sample.
    
    # Taking a look a the range of the EMG signals of the _3DCDataset, these seem to be 
    # integer values. Let's quickly get an idea of what may be the appropriate range to 
    # test for the WAMP threshold by inspecting the train_odh.
    derivatives = np.vstack([np.abs(np.diff(np.array(i),axis=0)) for i in train_odh.data[:]])

    plt.hist(derivatives, 100)
    plt.xlim(0, 4000)
    plt.show()

    # There only really are values from the 30-1000 range for the value of the derivative.
    # We will be counting the number of times the derivative of samples in a window exceed some value.
    # So, why don't we try WAMP thresholds from 0-2000

    # Lets initialize a grid search within that range.
    test_points = 500
    threshold_values = np.linspace(0, 2000, num=test_points)
    # And lets make a variable to store the accuracies for the test points
    threshold_results = np.zeros((test_points))

    # Begin computing the feature accuracies
    fe = libemg.feature_extractor.FeatureExtractor()
    om = libemg.offline_metrics.OfflineMetrics()

    for tp in range(test_points):
        dic = {
            "WAMP_threshold": float(threshold_values[tp])
        }
        train_features = fe.extract_features(["WAMP"], train_windows, dic)
        valid_features = fe.extract_features(["WAMP"], valid_windows, dic)
        model = libemg.emg_classifier.EMGClassifier()
        feature_dictionary = {"training_features": train_features,
                              "training_labels":   train_metadata["classes"]}
        try:
            # If we use a try block, we don't need to worry about non-invertable matrices
            # The results would just stay 0 as initialized.
            model.fit(model="LDA", feature_dictionary = feature_dictionary)
            predictions, probabilties = model.run(valid_features, valid_metadata["classes"])
            threshold_results[tp] = om.get_CA(valid_metadata["classes"], predictions) * 100
        finally:
            continue
    

    # plot the accuracies vs the thresholds
    plt.plot(threshold_values, threshold_results, marker='o', ms=3)
    best_threshold = np.argmax(threshold_results)
    plt.plot(threshold_values[best_threshold], threshold_results[best_threshold], marker="*", ms=5)
    plt.xlabel("WAMP Threshold")
    plt.ylabel("Accuracy (%)")
    plt.ylim((0, 100))
    plt.grid()
    plt.text(threshold_values[best_threshold],threshold_results[best_threshold], "({:.2f}, {:.2f})".format(threshold_values[best_threshold],threshold_results[best_threshold] ))
    plt.show()

    # Now let's apply this on our test set:
    dic = {
            "WAMP_threshold": float(threshold_values[best_threshold])
        }
    # Quick reminder: train_odh refers to the combined "train" and validation set
    train_windows, train_metadata = training_odh.parse_windows(50,10)
    test_windows, test_metadata   = test_odh.parse_windows(50,10)
    train_features = fe.extract_features(["WAMP"], train_windows, dic)
    test_features  = fe.extract_features(["WAMP"], test_windows, dic)
    feature_dictionary = {"training_features": train_features,
                          "training_labels":   train_metadata["classes"]}
    model = libemg.emg_classifier.EMGClassifier()
    model.fit(model="LDA", feature_dictionary = feature_dictionary)
    predictions, probabilties = model.run(test_features, test_metadata["classes"])
    test_accuracy = om.get_CA(test_metadata["classes"], predictions) * 100 

    print("Test accuracy with optimal WAMP threshold: {:.2f}%".format(test_accuracy))


