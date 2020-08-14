from numpy import load
import concatenated_classifier

# [1] Load Model
full_model = concatenated_classifier.load_model("Deception_Classifier_Model_Micros.h5")

# [2] Load Data
input_videos = load("test_videos.npy")
video_label = load("test_video_label.npy")
mapping = load("test_mapping_orig.npy")
micro_expressions = load("test_micro_exps.npy")

# [3] Pre-process data
input_videos, video_label, mapping, micro_expressions = concatenated_classifier.H1_preprocessing(input_videos, video_label,
                                                                                                 mapping, micro_expressions)

hist_eval = full_model.evaluate([input_videos, micro_expressions[0], micro_expressions[1],
                                 micro_expressions[2]], video_label, batch_size=8)
print(hist_eval)
# [4] Test Model
_, predictions = conc_c.test(full_model, input_videos, video_label, mapping, micro_expressions)
print(predictions)
# TODO: instead of printing access the predictions list with the index of the choosen video