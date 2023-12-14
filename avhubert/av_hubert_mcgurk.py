import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
import tempfile
from argparse import Namespace

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig

from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg


def install_requirements(): # TODO Sort out that
  # %cd /content/
  #!git clone https://github.com/facebookresearch/av_hubert.git
  #
  ## %cd av_hubert
  #!git submodule init
  #!git submodule update
  #!pip install scipy
  #!pip install sentencepiece
  #!pip install python_speech_features
  #!pip install scikit-video
  #
  ## %cd fairseq
  #!pip install ./
  ...


def install_preprocessing_tools():
  dir = "./tools"
  os.system(f'mkdir -p {dir}')
  os.system(f"wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O {os.path.join(dir, 'shape_predictor_68_face_landmarks.dat.bz2')}")
  os.system(f"bzip2 -d {os.path.join(dir, 'shape_predictor_68_face_landmarks.dat.bz2')}")
  os.system(f"wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O {os.path.join(dir, '20words_mean_face.npy')}")




def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(input_video_path, output_video_path, detector, predictor, mean_face_landmarks):
  STD_SIZE = (256, 256)
  stablePntsIDs = [33, 36, 39, 42, 45]
  videogen = skvideo.io.vread(input_video_path)
  frames = np.array([frame for frame in videogen])
  landmarks = []
  for frame in tqdm(frames):
      landmark = detect_landmark(frame, detector, predictor)
      landmarks.append(landmark)
  preprocessed_landmarks = landmarks_interpolate(landmarks)
  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                        window_margin=4, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  write_video_ffmpeg(rois, output_video_path, "/usr/bin/ffmpeg")
  return


def download_pretrained_model(model_url='https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/base_noise_pt_noise_ft_433h.pt', 
                              save_path='pretrained/finetune-model.pt'):
  os.system(f"wget {model_url} -O {save_path}")



def predict(video_path, audio_path, models, saved_cfg, task, tmp_dir):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{int(16_000*num_frames/30)}\n"]
  label_cont = ["DUMMY\n"]
  with open(f"{tmp_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{tmp_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  modalities = ["video", "audio"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  os.system("pwd")
  #models = [model.eval().cuda() for model in models]
  saved_cfg.task.modalities = modalities
  saved_cfg.task.data = tmp_dir
  saved_cfg.task.label_dir = tmp_dir

  #### MY OWN TESTS ####
  saved_cfg.task.noise_wav = tmp_dir
  saved_cfg.task.noise_snr = 1.0
  saved_cfg.task.noise_prob = 0.0
  #### HEHEHEHEHEHE ####

  task = tasks.setup_task(saved_cfg.task)
  task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  #sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo

def predict_video(origin_clip_path, mouth_roi_path, audio_path, detector, predictor, mean_face_landmarks, models, saved_cfg, task, tmp_dir):
  need_to_remove = False

  split = origin_clip_path.split('.')
  if split[-1] != 'mp4':
     ext_length  = len(split[-1])
     new_path = f"{origin_clip_path[:-(ext_length+1)]}.mp4"
     print(f"new_path = {new_path}")
     os.system(f"ffmpeg -hide_banner -loglevel error -i {origin_clip_path} {new_path}")
     origin_clip_path = new_path
     need_to_remove = True

  # isolate mouth
  preprocess_video(origin_clip_path, mouth_roi_path, detector, predictor, mean_face_landmarks)

  os.system(f'ffmpeg -hide_banner -loglevel error -i {origin_clip_path} -ac 1 -vn -ar 16000 {audio_path}')
  hypo = predict(mouth_roi_path, audio_path, models, saved_cfg, task, tmp_dir)

  if need_to_remove:
     os.system(f"rm {origin_clip_path}")
  os.system(f"rm {mouth_roi_path}")
  os.system(f"rm {audio_path}")

  return hypo

         

def control_experiment(dir):

  ## MAIN PIPELINE

  ckpt_path, face_predictor_path, mean_face_path = "pretrained/finetune-model.pt", "tools/shape_predictor_68_face_landmarks.dat", "tools/20words_mean_face.npy"
  data_path = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert/data"
  mouth_roi_path = os.path.join(data_path, "roi.mp4")
  audio_path = os.path.join(data_path, "output_audio.wav")
  user_dir = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert"
  tmp_dir = os.path.join(data_path, "tmp")


  # INSTALL STUFF WE NEED
  #install_preprocessing_tools()
  #download_pretrained_model()

  # models
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  mean_face_landmarks = np.load(mean_face_path)

  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  
  results = {}

  dir = os.path.join(data_path, dir)
  for video_name in sorted(os.listdir(dir)):
    video_path = os.path.join(dir, video_name)

    prediction = predict_video(video_path, mouth_roi_path, audio_path, detector, predictor, mean_face_landmarks, models, saved_cfg, task, tmp_dir)
    
    video_name = video_name.split('.')[0]

    print(f"\n\n==================================\n  PREDICTION({video_name}) : {prediction}\n")

    results[video_name] = prediction

  return results
     


def mc_gurk_experiment(experiments):

  ## MAIN PIPELINE

  ckpt_path, face_predictor_path, mean_face_path = "pretrained/finetune-model.pt", "tools/shape_predictor_68_face_landmarks.dat", "tools/20words_mean_face.npy"
  data_path = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert/data"
  mouth_roi_path = os.path.join(data_path, "roi.mp4")
  audio_path = os.path.join(data_path, "output_audio.wav")
  user_dir = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert"
  tmp_dir = os.path.join(data_path, "tmp")


  # INSTALL STUFF WE NEED
  #install_preprocessing_tools()
  #download_pretrained_model()

  # models
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  mean_face_landmarks = np.load(mean_face_path)

  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  
  results = {}

  # experiments is an array of 3-syllables tuples
  for auditory, visual, mg_expected in experiments:

    videos_path = os.path.join(data_path, 'mcgurk', f'{auditory}_{visual}_{mg_expected}')

    # store results
    predictions = []

    for video_name in sorted(os.listdir(videos_path)):
      origin_clip_path = os.path.join(videos_path, video_name)
      prediction = predict_video(origin_clip_path, mouth_roi_path, audio_path, detector, predictor, mean_face_landmarks, models, saved_cfg, task, tmp_dir)
    
      predictions.append(prediction)

    results[f'{auditory}_{visual}_{mg_expected}'] = predictions

  return results


def mc_gurk_word_experiment(videos_dir):

  ## MAIN PIPELINE

  ckpt_path, face_predictor_path, mean_face_path = "pretrained/finetune-model.pt", "tools/shape_predictor_68_face_landmarks.dat", "tools/20words_mean_face.npy"
  data_path = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert/data"
  mouth_roi_path = os.path.join(data_path, "roi.mp4")
  audio_path = os.path.join(data_path, "output_audio.wav")
  user_dir = "/home/talos/master_EPFL/ml/project_2/av_hubert/avhubert"
  tmp_dir = os.path.join(data_path, "tmp")


  # INSTALL STUFF WE NEED
  #install_preprocessing_tools()
  #download_pretrained_model()

  # models
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  mean_face_landmarks = np.load(mean_face_path)

  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  
  results = {}

  videos_dir = os.path.join(data_path, videos_dir)
  # experiments is an array of 3-syllables tuples
  for end_path in sorted(os.listdir(videos_dir)):
    video_path = os.path.join(videos_dir, end_path)

    prediction = predict_video(video_path, mouth_roi_path, audio_path, detector, predictor, mean_face_landmarks, models, saved_cfg, task, tmp_dir)

    results[end_path.split('.')[0]] = prediction

  return results