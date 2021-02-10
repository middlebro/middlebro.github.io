---
published: false
layout: post
title:  "[GitLab] How to setup"
categories: GitLab
tags: [gitlab]
comments: true
---

회사에 GitLab 서버를 설치하면서, 해당 방법을 정리해두면 좋을 것 같아 

   21  clear
   22  ll
   23  vi ~/.zshrc
   24  source ~/.zshrc
   25  clear
   26  cd ~
   27  cd gitlab_https
   28  ll
   29  rm server.csr
   30  rm server.key
   31  clear
   32  openssl version
   33  openssl genrsa -des3 -out gitlab.geosoft.co.kr.key 2048
   34  clear
   35  openssl req -new -key gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.csr
   36  openssl req -new -key gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.csr
   37  openssl x509 -req -days 3650 -in gitlab.geosoft.co.kr.csr -signkey gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.crt
   38  cp gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.key.ori
   39  openssl rsa -in gitlab.geosoft.co.kr.key.ori gitlab.geosoft.co.kr.key
   40  openssl rsa -in gitlab.geosoft.co.kr.key.ori -out gitlab.geosoft.co.kr.key
   41  clear
   42  openssl x509 -req -days 3650 -in gitlab.geosoft.co.kr.csr -signkey gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.crt
   43  cat gitlab.geosoft.co.kr.key | head -3
   44  cat gitlab.geosoft.co.kr.crt | head -3
   45  ll /etc/gitlab/ssl
   46  rm -rf /etc/gitlab/ssl
   47  sudo rm -rf /etc/gitlab/ssl
   48  mkdir -p /etc/gitlab/ssl
   49  sudo mkdir -p /etc/gitlab/ssl
   50  cd /etc/gitlab/ssl
   51  ll
   52  clear
   53  sudo cp ~/gitlab_https/gitlab.geosoft.co.kr.crt
   54  sudo cp ~/gitlab_https/gitlab.geosoft.co.kr.crt gitlab.geosoft.co.kr.crt
   55  sudo cp ~/gitlab_https/gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.key
   56  ll
   57  /etc/gitlab
   58  ll
   59  sudo vi gitlab.rb
   60  ssl
   61  ll
   62  sudo gitlab-ctl reconfigure
   63  ll
   64  sudo vi gitlab.rb
   65  cd ..
   66  sudo vi gitlab.rb
   67  sudo gitlab-ctl reconfigure
   68  cd ssl
   69  ll
   70  openssl req -new -days 365 -key gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.csr
   71  sudo openssl req -new -days 365 -key gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.csr
   72  cp gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.key.ori
   73  sudo cp gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.key.ori
   74  openssl rsa -in gitlab.geosoft.co.kr.key.ori -out gitlab.geosoft.co.kr.key
   75  sudo openssl rsa -in gitlab.geosoft.co.kr.key.ori -out gitlab.geosoft.co.kr.key
   76  openssl x509 -req -days 365 -in gitlab.geosoft.co.kr.csr -out gitlab.geosoft.co.kr.crt -signkey key
   77  openssl x509 -req -days 365 -in gitlab.geosoft.co.kr.csr -out gitlab.geosoft.co.kr.crt -signkey gitlab.geosoft.co.kr.key
   78  sudo openssl x509 -req -days 365 -in gitlab.geosoft.co.kr.csr -out gitlab.geosoft.co.kr.crt -signkey gitlab.geosoft.co.kr.key
   79  sudo firewall-cmd --add-service=https --permanent
   80  sudo gitlab-ctl reconfigure
   81  sudo openssl req -new -days 365 -key gitlab.geosoft.co.kr.key -out gitlab.geosoft.co.kr.csr
   82  sudo cp gitlab.geosoft.co.kr.key gitlab.geosoft.co.kr.key.ori
   83  sudo openssl x509 -req -days 365 -in gitlab.geosoft.co.kr.csr -out gitlab.geosoft.co.kr.crt -signkey gitlab.geosoft.co.kr.key
   84  sudo vi /etc/gitlab/gitlab.rb
   85  sudo gitlab-ctl reconfigure
   86  sudo vi /etc/gitlab/gitlab.rb
   87  sudo gitlab-ctl reconfigure
   88  clear
   89  sudo vi /etc/gitlab/gitlab.rb
   90  sudo gitlab-ctl reconfigure
   91  sudo vi /etc/gitlab/gitlab.rb
   92  sudo gitlab-ctl reconfigure
   93  ps
   94  ps -ax
   95  ps -ax | grep apache
   96  service apache stop
   97  systemctl status httpd
   98  systemctl status apache2.service
   99  systemctl stop apache2.service
  100  vi /etc/gitlab/gitlab.rb
  101  sudo vi /etc/gitlab/gitlab.rb
  102  sudo gitlab-ctl reconfigure
  103  cd ..
  104  rm -rf gitlab
  105  ll
  106  rm -rf ssl
  107  sudo rm -rf ssl
  108  mkdir ssl
  109  sudo mkdir ssl
  110  clear
  111  vi /etc/gitlab/gitlab.rb
  112  sudo vi /etc/gitlab/gitlab.rb
  113  sudo gitlab-ctl reconfigure
  114  sudo vi /etc/gitlab/gitlab.rb
  115  sudo gitlab-ctl reconfigure
  116  shutdown now
  117  sudo shutdown now
  118  ll
  119  ssh 218.153.121.56
  120  clear
  121  ll
  122  curl http://gitlab.geosoft.co.kr
  123  clear
  124  service stop apache2
  125  systemctl stop apache2.service
  126  clear
  127  systemctl stop apache2.service
  128  /etc/gitlab
  129  ll
  130  sudo vi gitlab.rb
  131  sudo gitlab-ctl reconfigure
  132  sudo gitlab-rails console
  133  ll
  134  pwd
  135  netstat -ntpl
  136  /var/opt/gitlab/gitlab-rails/etc
  137  sudo cd /var/opt/gitlab/gitlab-rails/etc
  138  sudo su
  139  ll
  140  pwd
  141  history
  142  sudo vim /var/opt/gitlab/gitlab-rails/etc/gitlab.yml
  143  sudo yum install rsync
  144  cat /etc/issue
  145  sudo apt-get install rsync
  146  gitlab-ctl version
  147  vi /etc/fstab
  148  ll
  149  echo "Hello World" > hello.txt
  150  ll
  151  cat hello.txt
  152  echo "Hello World" > hello.txt
  153  cat hello.txt
  154  echo "Hello World" >> hello.txt
  155  cat hello.txt
  156  echo "Hello WorldHello World!" >> hello.txt
  157  cat hello.txt
  158  echo -e "Hello WorldWorld!" >> hello.txt
  159  echo -e "Hello World! \n Hello World >> hello.txt\n"
  160  cat hello.txt
  161  echo "Hello World"
  162  echo -e "Hello\nWorld"
  163  echo -e "Hello\nWorld" >> hello.txt
  164  ll
  165  cat hello.txt
  166  rm hello.txt
  167  ll
  168  vi /etc/fstab
  169  rsync -V
  170  rsync 00version
  171  rsync --version
  172  clear
  173  rsync --version
  174  vi /etc/gitlab/gitlab.rb
  175  sudo vi /etc/gitlab/gitlab.rb
  176  sudo su - crontab -e
  177  sudo su - crontab - e
  178  sudo su - crontab
  179  sudo su
  180  crontab -e
  181  sudo su - crontab -e
  182  crontab -e
  183  cat /etc/issue
  184  apt-get update
  185  sudo apt-get update
  186  sudo apt-get upgrade
  187  apt-get install mysql-server
  188  sudo apt-get install mysql-server
  189  lsb_release -a
  190  sudo apt-get install mysql-server
  191  sudo systemctl enable mysql
  192  locale
  193  localectl status
  194  locale -a
  195  sudo dpkg-reconfigure locales
  196  cat /etc/default/locale
  197  sudo update-locale LANG=ko_KR.UTF-8
  198  locale -a
  199  sudo apt install language-pack-ko
  200  sudo dpkg-reconfigure locales
  201  sudo vi /etc/environment
  202  echo $ENV
  203  echo $PATH
  204  echo $LANG
  205  echo $LC_ALL
  206  LC_ALL=en_US.UTF-8
  207  echo $LC_ALL
  208  sudo systemctl enable mysql
  209  sudo systemctl status mysql
  210  mysql_secure_installation
  211  sudo mysql_secure_installation
  212  mysql -uroot -p
  213  sudo mysql -uroot -p
  214  sudo apt-get install ruby-full
  215  ruby -V
  216  ruby --version
  217  sudo apt-get install nginx -y
  218  systemctl status nginx
  219  sudo systemctl status nginx
  220  sudo apt-get install dirmngr gnupg apt-transport-https ca-certificates
  221  apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 561F9B9CAC40B2F7
  222  sudo apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 561F9B9CAC40B2F7D
  223  add-apt-repository 'deb https://oss-binaries.phusionpassenger.com/apt/passenger bionic main'
  224  sudo add-apt-repository 'deb https://oss-binaries.phusionpassenger.com/apt/passenger bionic main'
  225  sudo apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 561F9B9CAC40B2F7D
  226  apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 561F9B9CAC40B2F7D
  227  apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com 561F9B9CAC40B2F7D
  228  apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com 561F9B9CAC40B2F7
  229  gpg --keyserver hkp://keyserver.ubuntu.com --recv-keys 843938DF228D22F7B3742BC0D94AA3F0EFE21092 C5986B4F1257FFA86632CBA746181433FBB75451
  230  add-apt-repository 'deb https://oss-binaries.phusionpassenger.com/apt/passenger bionic main'
  231  sudo add-apt-repository 'deb https://oss-binaries.phusionpassenger.com/apt/passenger bionic main'
  232  sudo apt-get install apache2 libapache2-mod-passenger
  233  sudo systemctl start apache2
  234  sudo systemctl start apache2.service
  235  journalctl -xe
  236  sudo journalctl -xe
  237  systemctl
  238  sudo vi /etc/apache2/ports.conf
  239  sudo systemctl start apache2
  240  systemctl
  241  systemctl status apacheche2.service
  242  systemctl status apache2.service
  243  sudo vi /etc/apache2/ports.conf
  244  ip
  245  hostname -i
  246  curl ifconfig.me
  247  sudo apt-get install redmine redmine-mysql -y
  248  sudo gem update
  249  sudo gem install bundler
  250  sudo vi /etc/apache2/mods-available/passenger.conf
  251  apache2 --version
  252  apache2 -V
  253  sudo vi /etc/apache2/sites-available/000-default.conf
  254  sudo touch /usr/share/redmine/Gemfile.lock
  255  sudo chown www-data:www-data /usr/share/redmine/Gemfile.lock
  256  sudo systemctl restart apache2
  257  curl -O http://localhost:8081/redmine
  258  curl  http://localhost:8081/redmine
  259  curl  http://localhost:80/redmine
  260  sudo vi /etc/apache2/sites-available/000-default.conf
  261  sudo systemctl restart apache2
  262  systemctl status apache2.service
  263  /etc/apache2/sites-available
  264  ll
  265  vi 000-default.conf
  266  cd ~
  267  sudo vi /etc/apache2/ports.conf
  268  sudo vi /etc/apache2/sites-available/000-default.conf
  269  /var/www/html/redmine
  270  ll
  271  tree .
  272  sudo chown www-data:www-data /usr/share/redmine/Gemfile.lock
  273  ll
  274  mysql -uroot -p
  275  sudo mysql -uroot -p
  276  /
  277  ~
  278  clear
  279  ll
  280  sudo apt-get install apache2 libapache2-mod-passenger
  281  sudo apt-get install mysql-server mysql-client
  282  sudo apt-get install redmine redmine-mysql
  283  sudo gem update
  284  sudo gem install bundler
  285  sudo vi /etc/apache2/mods-available/passenger.conf
  286  sudo vi /etc/apache2/sites-available/000-default.conf
  287  sudo systemctl restart apache2
  288  sudo vi /etc/apache2/sites-available/000-default.conf
  289  sudo systemctl restart apache2
  290  sudo vi /etc/apache2/sites-available/000-default.conf
  291  sudo systemctl restart apache2
  292  sudo vi /etc/apache2/sites-available/000-default.conf
  293  sudo systemctl restart apache2
  294  \/usr/share/redmine/public
  295  /usr/share/redmine/public
  296  ll
  297  tree .
  298  ~
  299  sudo gem update
  300  sudo systemctl restart apache2
  301  su
  302  sudo su - redmine
  303  sudo apt-get install svn
  304  wget https://www.redmine.org/releases/redmine-4.1.1.tar.gz
  305  ll
  306  tar -xvf redmine-4.1.1.tar.gz
  307  ll
  308  cd redmine-4.1.1
  309  ll
  310  config
  311  cp database.yml.example database.yml
  312  sudo vi database.yml
  313  cd ~
  314  sudo vi vi /etc/apache2/mods-available/passenger.conf
  315  sudo vi /etc/apache2/mods-available/passenger.conf
  316  sudo systemctl restart apache2
  317  sudo vi /etc/apache2/mods-available/passenger.conf
  318  sudo apt-get install postfix
  319  sudo vi /etc/redmine/default/configuration.yml
  320  sudo systemctl status postfix
  321  sudo systemctl restart apache2
  322  sudo apt-get install git subversion cvs mercurial
  323  sudo service apache2 restart
  324  ll
  325  ssh 218.153.121.56
  326  ll
  327  redmine
  328  ll
  329  find / -name redmine
  330  cd /usr/share/redmine
  331  ll
  332  cd ~
  333  sudo find / -name passenger.conf
  334  vi /etc/apache2/mods-enabled/passenger.conf
  335  cat /usr/lib/ruby/vendor_ruby/phusion_passenger/locations.ini
  336  cd /etc/gitlab
  337  ll
  338  cat gitlab.rb
  339  vi gitlab.rb
  340  sudo vi gitlab.rb
  341  sudo gitlab-ctl reconfigure
  342  gitlab-ctl status
  343  sudo gitlab-ctl status