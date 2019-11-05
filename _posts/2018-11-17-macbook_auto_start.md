---
layout: post
title: 맥북 자동 부팅 옵션 설정법
category: [Living]
tags: [Macbook]
sitemap :
changefreq : daily
---

안녕하세요! Jerry 입니다!

오늘은 맥북 자동 부팅 옵션에 대해서 설정법을 알아보겠습니다.

매우 짧은 포스팅이 될 듯합니다.

1. Terminal 을 실행한다.
2. `sudo nvram AutoBoot=%00` 이라고 친다.
3. 맥북 패스워드를 입력한다.
4. 전원을 끈 후 노트북을 닫고 열어서 확인한다.

끝입니다.. 이러면 자동 부팅 옵션이 꺼집니다.

만약 다시 자동 부팅을 켜시려면 1번을 실행하신 후 `sudo nvram AutoBoot=%03` 이라고 입력하시면 됩니다.

감사합니다!
