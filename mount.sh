sudo mkdir -p /mnt/share
sudo mount -t cifs //10.20.30.8/share /mnt/share -o guest,file_mode=0777,dir_mode=0777
ls /mnt/share/

