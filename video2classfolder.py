import numpy as np
import os
import shutil

root = r'.'
video_folder = r'diving_samples_len_151_lstm'
class_folder = r'byclass'
score_file = r'diving_overall_scores.csv'

# scores = pd.read_csv('diving_overall_scores.csv')
scores = np.genfromtxt(score_file, delimiter=",")

video_path = os.path.join(root, video_folder)

for index, video in enumerate(os.listdir(video_path)):
    print(str(index) + ": " + video)
    print(scores[index])
    src = os.path.join(video_path, video)
    if scores[index] >= 90:
        dst = os.path.join(root, class_folder, "90")
        dst = os.path.join(dst, video)
        shutil.copyfile(src,dst)
    elif scores[index] >= 80:
        dst = os.path.join(root, class_folder, "80")
        dst = os.path.join(dst, video)
        shutil.copyfile(src,dst)
    elif scores[index] >= 70:
        dst = os.path.join(root, class_folder, "70")
        dst = os.path.join(dst, video)
        shutil.copyfile(src,dst)
    elif scores[index] >= 60:
        dst = os.path.join(root, class_folder, "60")
        dst = os.path.join(dst, video)
        shutil.copyfile(src,dst)
    else:
        print(video + ": less than 60") 
    print('='*50)
#     if index%100 == 99:
#         break
