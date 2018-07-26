# Setup AWS resources

1. Turn on EC2 p2.xlarge spot instance, 1 GPU, 4 vCPUs, 61 RAM (GiB), 400 GB SSD
2. using Deep Learning AMI (Ubuntu) Version 4.0 (ami-d884b1bd)
3. allow ibound traffic on ports 8081-8083 and 6006-6009
4. Install docker

```
apt install docker.io
```

5. attach EFS volume in same security group as

```
apt-get install nfs-common
mkdir /mnt/knowdnn_hdd
mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-dcc732a5.efs.us-east-2.amazonaws.com:/ /mnt/knowdnn_hdd
```

# Set Up User Cloud9s

0. Set user environment variables:

```
C9PASS='tmppass'
```

1. Casey:

```
USERID='crhanso2'
UPORT='8081'
docker run -d --restart=always --privileged \
    --name c9-$USERID -h c9-$USERID -p $UPORT:8181 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(which docker):$(which docker) \
    -v /home/$USERID:/workspace \
    -v /home/tfuser:/workspace/home/tfuser \
    -v /mnt/knowdnn_hdd/tfuser:/workspace/mnt/knowdnn_hdd/tfuser \
    cblatti3/tf_cloud9:1.0 --auth $USERID:$C9PASS
```

2. Bryce:

```
USERID='kille2'
UPORT='8082'
docker run -d --restart=always --privileged \
    --name c9-$USERID -h c9-$USERID -p $UPORT:8181 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(which docker):$(which docker) \
    -v /home/$USERID:/workspace \
    -v /home/tfuser:/workspace/home/tfuser \
    -v /mnt/knowdnn_hdd/tfuser:/workspace/mnt/knowdnn_hdd/tfuser \
    cblatti3/tf_cloud9:1.0 --auth $USERID:$C9PASS
```

3. Charles:

```
USERID='blatti'
UPORT='8083'
docker run -d --restart=always --privileged \
    --name c9-$USERID -h c9-$USERID -p $UPORT:8181 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(which docker):$(which docker) \
    -v /home/$USERID:/workspace \
    -v /home/:/workspace/home/ \
    -v /mnt/:/workspace/mnt/ \
    cblatti3/tf_cloud9:1.0 --auth $USERID:$C9PASS
```

# Setup External Directories

1. Map locations in cloud9:

```
ln -s /workspace/home/tfuser/ /home/
mkdir -p /mnt/
ln -s /workspace/mnt/knowdnn_hdd/ /mnt/
mkdir ~/.ssh/
ln -s /workspace/home/tfuser/creds/ssh_config ~/.ssh/config
```

# Activate Tensorflow

1. When logged in as root

```
source activate tensorflow_p36
```