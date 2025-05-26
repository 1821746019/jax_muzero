#! /bin/bash
# gym新版和jax_muzero不兼容，gym[atari]==0.17.3没有内置breakout游戏的ROM，需要手动下载并导入
file_path=/tmp/ROMS
extract_path=/tmp/ROMS
wget http://www.atarimania.com/roms/Roms.rar -O $file_path
apt install unrar -y
unrar x $file_path $extract_path
#导入
python -m atari_py.import_roms $extract_path

#删除临时文件
rm -rf $file_path
rm -rf $extract_path