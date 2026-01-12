#!/bin/bash
rsync -avz \
  --exclude='__pycache__' \
  --exclude='.history' \
  --exclude='.git' \
  --exclude='**/play.py' \
  --exclude='deploy' \
  yulong@131.159.60.67:/home/yulong/jiabao_song/cpg_jump/ \
  /home/song/cpg_jump/
