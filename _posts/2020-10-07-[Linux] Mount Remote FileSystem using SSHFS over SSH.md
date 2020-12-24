---
published: true
layout: post
title:  "[Linux] SSHFS를 통한 외부 저장소 마운트"
categories: [Linux, GitLab, SSHFS]
tags: [gitLab, crontab, sshfs]
comments: true
---
 
리눅스를 다루다 보면 특정 디렉토리를 다른 서버의 디렉토리와 연결하여 사용하고 싶을 때가 종종 발생한다. 예를 들면 서버 파일의 백업 혹은 외부 편집기를 통한 파일 수정 등.
 
마운트 하는 방법에는 여러 방법들이 있지만, 연결하고 싶은 컴퓨터에 SSH 가 열려있다면 SSHFS를 이용하여 간단히 마운트할 수 있다. SSHFS(Secure SHell FileSystem)는 SFTP로 파일을 전송하는 클라이언트다.

### SSHFS 설치
SSHFS 는 각 시스템에서 지원하는 package manager로 간단하게 설치할 수 있으니, 자기 환경에 맞는 명령어를 통해 설치해주면 된다.

```bash
$ sudo apt-get install sshfs
$ sudo yum install sshfs
$ sudo dnf install sshfs
```

### 마운트할 경로 생성
보통 마운트 경로는 /mnt 밑에 생성하므로 다음과 같이 디렉토리를 생성 해준다.
```bash
$ sudo mkdir -p /mnt/backups 
```

### SSHFS를 이용한 연결(mount)
```bash
$ sshfs [user@]hostname:[directory] mountpoint

$ # username이 user이고 ip 주소가 127.0.0.1인 
$ # 백업서버의 /home/user/backup 를 /mnt/backups 에 마운트 한다고 가정하면
$ sshfs user@127.0.0.1:/home/user/backup /mnt/backups
```

연결이 끊기면 재접속이 가능하도록 다음과 같이 설정해 줄 수 있다.
```bash
$ sshfs -o reconnect [user@]hostname:[directory] mountpoint

$ # SSH key 인증 방식을 쓴다면 인증 파일을 지정해 줄 수도 있다.
$ sshfs -o IdentityFile='KEY_PATH' [user@]hostname:[directory] mountpoint
```

### Mount 확인
`df` 명령으로 마운팅된 경로와 해당 서버 디렉토리의 경로를 알 수 있다.

``` bash
$ df -hT
Filesystem                          Type        Size  Used Avail Use% Mounted on
udev                                devtmpfs    730M     0  730M   0% /dev
tmpfs                               tmpfs       150M  4.9M  145M   4% /run
/dev/sda1                           ext4         31G  5.5G   24G  19% /
tmpfs                               tmpfs       749M  216K  748M   1% /dev/shm
tmpfs                               tmpfs       5.0M  4.0K  5.0M   1% /run/lock
tmpfs                               tmpfs       749M     0  749M   0% /sys/fs/cgroup
tmpfs                               tmpfs       150M   44K  150M   1% /run/user/1000
**tecmint@192.168.0.102:/home/tecmint fuse.sshfs  324G   55G  253G  18% /mnt/tecmint**
```

서버가 재시작되어도 항상 연결되어 있도록 하기 위해 다음 명령어를 통해 `/etc/fstab`을 수정해준다.

```bash
$ echo "sshfs#[user@]hostname:[directory] mountpoint
fuse.sshfs defaults 0 0" >> /etc/fstab

# 필요한 옵션을 추가해 줄 수도 있다.
sshfs#[user@]hostname:[directory] mountpoint fuse.sshfs IdentityFile=~/.ssh/id_rsa defaults 0 0
sshfs#[user@]hostname:[directory] mountpoint fuse.sshfs reconnect defaults 0 0
```

### 마운트 해제
연결을 끊고 싶으면 `unmount` 명령을 사용한다.

```bash
$ unmount /mnt/backups
```

만약 `/etc/fstab` 파일을 수정했다면 추가한 내용을 삭제해준다.


## Reference
- [How to Mount Remote Linux Filesystem or Directory Using SSHFS Over SSH]

[GitLab 공식문서]: https://docs.gitlab.com/ee/raketasks/backup_restore.html#restore-gitlab
[How to Mount Remote Linux Filesystem or Directory Using SSHFS Over SSH]: https://www.tecmint.com/sshfs-mount-remote-linux-filesystem-directory-using-ssh/

> Written with [StackEdit](https://stackedit.io/).
