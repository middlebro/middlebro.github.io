---
published: false
layout: post
title:  "[PostgreSQL] 대용량 서비스를 위한 클러스터 구성하기"
categories: PostgreSQL
tags: [PostgresSQL, Cluster]
comments: true
---

PostgreSQL을 클러스터 구성하는 방법에 대해 알아보겠습니다.

본 포스트는 postgresql v12, CentOS8 환경에서 작성되었습니다.

## Intro
서비스가 대형화 될 수록 다량의 트래픽이 발생하게 되고, 서버 하나로 처리할 수 있는 량은 정해져 있기 때문에, 분산 서버를 구축하고 클러스터링 합니다. 
액티브 서버와 복제 서버는 동시에 가동하며, 복제 서버는 select 기능만 가능하도록 구성해보겠습니다.

## 1. 액티브 서버 구성
### 설치
```
dnf module list postgresql
dnf module enable postgresql:12
dnf install postgresql-server
```

### DB 초기화
```
postgresql-setup --initdb
```


### 서비스 등록 및 시작
```
systemctl start postgresql
systemctl enable postgresql
```

### 방화벽 설정
```
firewall-cmd --permanent --zone=public --add-port=postgresql/tcp
firewall-cmd --reload
```

### postgres 계정전환
```
sudo -i -u postgres
```

### replicator 계정생성, 패스워드 설정
```
createuser --replication -P -e replicator
```

### 접속정보 수정, postgres.auto.conf 파일에 자동적용됨.
```
psql -c "ALTER SYSTEM SET listen_addresses TO '*';"
```

### 계정 패스워드 수정 및 계정목록 확인
```
psql
\password postgres

\du
```



### replication 추가
```
vi /var/lib/pgsql/data/pg_hba.conf
host    replication     replicator      192.168.101.0/24        md5
host    all     		all      		192.168.101.0/24        md5
```

### DB 재시작, postgres 계정에서 재시작 되지 않을 경우 root 계정으로 재시작
```
systemctl restart postgresql.service
```



## 2. 복제 서버 구성
### 설치
```
dnf module list postgresql
dnf module enable postgresql:12
dnf install postgresql-server
```

### 방화벽 설정
```
firewall-cmd --permanent --zone=public --add-port=postgresql/tcp
firewall-cmd --reload
```


### postgres 계정전환
```
sudo -i -u postgres
```

### data 디렉토리 백업처리
```
mv data data_backup
```

### 동기화
```
pg_basebackup -h 192.168.101.105 -D /var/lib/pgsql/data -U replicator -P -v  -R -X stream -C -S pg_standby_1
```

### 설정 확인
```
cat /var/lib/pgsql/data/postgresql.auto.conf
```
### 액티브 서버 계정에서 pg_replication_slots 생성확인, pg_standby_1 유무 체크
```
psql -c "SELECT * FROM pg_replication_slots;"
```

### 설정 확인, 정상일 경우 primary_conninfo 내용이 확인됨
```
cat /var/lib/pgsql/data/postgresql.auto.conf
```
### 서비스 등록 시작
```
systemctl start postgresql
systemctl enable postgresql
```

### 복제 서버쪽에서 확인, walsender replicator 항목을 확인
```
psql -c "\x" -c "SELECT * FROM pg_stat_wal_receiver;"
```
### 액티브 서버 쪽에서 확인
```
psql -c "\x" -c "SELECT * FROM pg_stat_replication;"
```
### 액티브 서버에서 복제 서버 연결 확인
```
systemctl status postgresql
```
### 연동확인, 액티브 서버에서 DB 생성
```
psql
CREATE DATABASE test_db;

\l
```

### 복제 서버에서 확인
```
psql
\l
```
정상적으로 완료된 경우라면 액티브 쪽에서 생성된 DB를 복제 서버에서도 동일하게 확인할 수 있습니다.


## 3. 동기복제(비동기에서 동기로 수정하는 방법)
### 액티브 서버 쪽에서 복제 서버 이름을 지정하면 해당하는 스탠바이 서버와 복제가 일어난다.
```
psql -c "ALTER SYSTEM SET synchronous_standby_names TO '*';"
```
### 서비스 재시작
```
systemctl restart postgresql
```


