---
published: true
layout: post
title:  "[GitLab] Backup with Crontab and SSHFS"
categories: GitLab
tags: [gitlab, backup, crontab, sshfs]
comments: true
---

이전에 구현했던 GitLab 서버에 백업기능을 추가해보자.

## 요구사항
먼저 GitLab의 백업기능을 사용하기 위해 `rsync`가 설치되어 있는지 확인한다.

다음 명령어를 통해 설치 여부를 확인한다.

``` console
$ rsync --version

Copyright (C) 1996-2015 by Andrew Tridgell, Wayne Davison, and others.
Web site: http://rsync.samba.org/
Capabilities:
    64-bit files, 64-bit inums, 64-bit timestamps, 64-bit long ints,
    socketpairs, hardlinks, symlinks, IPv6, batchfiles, inplace,
    append, ACLs, xattrs, iconv, symtimes, prealloc

rsync comes with ABSOLUTELY NO WARRANTY.  This is free software, and you
are welcome to redistribute it under certain conditions.  See the GNU
General Public Licence for details.
```

만약 `rsync`가 설치되어 있지 않다면, 시스템 환경에 알맞는 명령어를 통해 설치해준다.

``` bash
$ # Debian/Ubuntu
$ sudo apt-get install rsync

$ # RHEL/CentOS
$ sudo yum install rsync
```

## 외부 저장소 마운트

나는 백업파일을 개인서버에 저장할 것이기 때문에 SSHFS를 이용하여 먼저 백업서버를 마운트 해보도록 하겠다.

다음 명령어를 통해 SSHFS 를 설치해주고, 백업서버를 마운트할 디렉토리를 생성해준다.
``` bash
$ sudo apt-get install sshfs
$ sudo mkdir -p /mnt/backups
```

연결할 BackUp 서버의 username hostname과 directory 를 알맞게 지정한 뒤 위에서 생성한 `/mnt/backups`에 마운트해준다.
```console
$ sshfs user@127.0.0.1:/home/user/backup /mnt/backups
```

## GitLab 백업

GitLab서버의 백업 위치를 마운트된 백업서버로 변경해주기 위해 `/etc/gitlab/gitlab.rb`를 수정해준다.

``` ruby
gitlab_rails['backup_upload_connection']  =  {
  :provider  =>  'Local',
  :local_root  =>  '/mnt/backups' 
}
# The directory inside the mounted folder to copy backups to
# Use '.' to store them in the root directory  
gitlab_rails['backup_upload_remote_directory']  =  'gitlab_backups'
```

백업 파일 권한도 설정해 줍니다.
```
gitlab_rails['backup_archive_permissions']  =  0644  # Makes the backup archives world-readable
```

마지막으로 `crontab`을 편집하여 gitlab-backup을 스케쥴링 해준다.
``` bash
$ sudo crontab -e

      1 # Edit this file to introduce tasks to be run by cron.
      2 #
      3 # Each task to run has to be defined through a single line
      4 # indicating with different fields when the task will be run
      5 # and what command to run for the task
      6 #
      7 # To define the time you can provide concrete values for
      8 # minute (m), hour (h), day of month (dom), month (mon),
      9 # and day of week (dow) or use '*' in these fields (for 'any').#
     10 # Notice that tasks will be started based on the cron's system
     11 # daemon's notion of time and timezones.
     12 #
     13 # Output of the crontab jobs (including errors) is sent through
     14 # email to the user the crontab file belongs to (unless redirected).
     15 #
     16 # For example, you can run a backup of all your user accounts
     17 # at 5 a.m every week with:
     18 # 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
     19 #
     20 # For more information see the manual pages of crontab(5) and cron(8)
     21 #
     22 # m h  dom mon dow   command
     23 0 2 * * * /opt/gitlab/bin/gitlab-backup create CRON=1

```

## 백업 수명 제한

디스크 공간을 절약하기 위해 오래된 백업파일에 대해 수명제한을 걸 수 있다.

다음과 같이 `/etc/gitlab/gitlab.rb`파일을 수정해준다.
``` bash
## Limit backup lifetime to 7 days - 604800 seconds  
gitlab_rails['backup_keep_time']  =  604800
```
위에 나오는 `604800`은 초 단위라는 것을 유의하여 해당 값을 적절하게 수정해주면 된다.


> Written with [StackEdit](https://stackedit.io/).

[GitLab 공식문서]: https://docs.gitlab.com/ee/raketasks/backup_restore.html#restore-gitlab

[SSHFS]: https://BackUp
