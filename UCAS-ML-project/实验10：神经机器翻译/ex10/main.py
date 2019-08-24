"""The entry of the nmt model"""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

import nmt
import inference
import train
from utils import evaluation_utils
from utils import misc_utils as utils
from utils import vocab_utils


if __name__ == "__main__":
  # 读入命令行参数并修改 parser argument
  # 这里我们几乎已经设置好 default 值了，所以应该无需输入命令行参数
  nmt_parser = argparse.ArgumentParser(conflict_handler='resolve')
  nmt.add_arguments(nmt_parser)
  
  nmt.FLAGS, unparsed = nmt_parser.parse_known_args()
  nmt.FLAGS.out_dir = "./output/nmt_attention_model"

  # train and dev
  print("-------- [Info] Start train and dev --------")
  tf.app.run(main=nmt.main, argv=[sys.argv[0]] + unparsed)
  print("-------- [Info] End train and dev --------")
  
  '''
  # inference/translate
  nmt.FLAGS.inference_input_file = "./output/my_infer_file.vi"
  nmt.FLAGS.inference_output_file = "./output/nmt_attention_model/output_infer"

  # train and dev
  print("-------- [Info] Start inference --------")
  tf.app.run(main=nmt.main, argv=[sys.argv[0]] + unparsed)
  print("-------- [Info] End inference --------")
  '''
  