---
published: true
title: [macOS] iTerm2 한글 자소 분리 해결
category: [macOS]
use_math: true
---

포스트 글을 작성하던 중에 현재 작업 중인 디렉토리에 들어갈 일이 생겨서 터미널을 열어보니
다음과 같이 파일 이름이 표시되는 경우가 발생하였다.

![img1](/images/macos/202009240001.png)

## 문제 원인

* 모든 글자는 `유니코드`라는 산업 표준에 따라 표현하고 다루는데,
Windows와 macOS 등 서로 다른 방식으로 처리하기 때문
* Windows에서는 자소처리를 `NFC` 방식으로 처리
* macOS에서는 자소처리를 `NFD` 방식으로 처리

>**NFC (Normalize Form C)**
>
>NFC는 모든 음절을 Canonical Decomposition(정준 분해) 후 Canonical Composition(정준 결합) 하는 방식이다. 즉, 각을 각이라는 하나의 문자로 저장하는 방식이다. 이 방식을 사용하면 NFD 방식보다 텍스트의 사이즈는 작아지게 된다. 하지만, 옛 한글 자모의 결합으로 이루어진 한글 음절 코드가 없으므로 이 음절은 Canonical Composition 하지 못하므로 자소가 분리된 체로 저장하게 된다. 이로 인해, 현대 한글과 옛 한글이 다른 방식으로 저장되므로 텍스트를 처리할 때 유의해야 한다.
>
>**NFD (Normalize Form D)**
>
>NFD는 모든 음절을 Canonical Decomposition(정준 분해)하여 한글 자모 코드를 이용하여 저장하는 방식이다. 즉, 각을 ㄱ + ㅏ + ㄱ 로 저장하는 방식이다. 이 방식은 현대 한글과 옛 한글을 동일한 방식으로 저장한다는 장점이 있지만 NFC 방식과 비교하여 텍스트의 크기가 커진다는 문제가 있다.

## 문제 해결 방법

### iTerm2

![img2](/images/macos/202009240002.png)
![img3](/images/macos/202009240003.png)

* Preferences > Profiles > Text > Unicode > Unicode normalization form
Unicode normalization form의 값으로 NFC를 선택

### iTerm2 화면 확인

![img4](/images/macos/202009240004.png)

## 참고

* [한글과 유니코드 - Pusnow](https://gist.github.com/Pusnow/aa865fa21f9557fa58d691a8b79f8a6d)
* [[MAC] macOS iTerm2 한글 자소분리 현상 해결](https://velog.io/@inyong_pang/MAC-macOS-iTerm2-한글-자소분리-현상-해결)