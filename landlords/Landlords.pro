QT       += core gui multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    animationwindow.cpp \
    bgmcontrol.cpp \
    buttongroup.cpp \
    card.cpp \
    cardpanel.cpp \
    cards.cpp \
    countdown.cpp \
    endingpanel.cpp \
    gamecontrol.cpp \
    loading.cpp \
    main.cpp \
    gamepanel.cpp \
    mybutton.cpp \
    player.cpp \
    playhand.cpp \
    robot.cpp \
    robotgraplord.cpp \
    robotplayhand.cpp \
    scorepanel.cpp \
    strategy.cpp \
    userplayer.cpp

HEADERS += \
    animationwindow.h \
    bgmcontrol.h \
    buttongroup.h \
    card.h \
    cardpanel.h \
    cards.h \
    countdown.h \
    endingpanel.h \
    gamecontrol.h \
    gamepanel.h \
    loading.h \
    mybutton.h \
    player.h \
    playhand.h \
    robot.h \
    robotgraplord.h \
    robotplayhand.h \
    scorepanel.h \
    strategy.h \
    userplayer.h

FORMS += \
    buttongroup.ui \
    gamepanel.ui \
    scorepanel.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    res.qrc

RC_ICONS = logo.ico

DISTFILES += \
    conf/playList.json \
    images/1fen-1.png \
    images/1fen-2.png \
    images/1fen-3.png \
    images/2fen-1.png \
    images/2fen-2.png \
    images/2fen-3.png \
    images/3fen-1.png \
    images/3fen-2.png \
    images/3fen-3.png \
    images/background-1.png \
    images/background-2.png \
    images/background-3.png \
    images/background-4.png \
    images/background-5.png \
    images/background-6.png \
    images/background-7.png \
    images/background-8.png \
    images/background-9.png \
    images/background-10.png \
    images/bomb_1.png \
    images/bomb_2.png \
    images/bomb_3.png \
    images/bomb_4.png \
    images/bomb_5.png \
    images/bomb_6.png \
    images/bomb_7.png \
    images/bomb_8.png \
    images/bomb_9.png \
    images/bomb_10.png \
    images/bomb_11.png \
    images/bomb_12.png \
    images/buqiang-1.png \
    images/buqiang-2.png \
    images/buqiang-3.png \
    images/buqinag.png \
    images/button_hover.png \
    images/button_normal.png \
    images/button_pressed.png \
    images/card.png \
    images/chupai_btn-1.png \
    images/chupai_btn-2.png \
    images/chupai_btn-3.png \
    images/clock.png \
    images/farmer_fail.png \
    images/farmer_man_1.png \
    images/farmer_man_2.png \
    images/farmer_win.png \
    images/farmer_woman_1.png \
    images/farmer_woman_2.png \
    images/farmer-right.png \
    images/fen.png \
    images/gameover.png \
    images/jianhao.png \
    images/jiaodizhu.png \
    images/joker_bomb_1.png \
    images/joker_bomb_2.png \
    images/joker_bomb_3.png \
    images/joker_bomb_4.png \
    images/joker_bomb_5.png \
    images/joker_bomb_6.png \
    images/joker_bomb_7.png \
    images/joker_bomb_8.png \
    images/liandui.png \
    images/loading.png \
    images/lord_fail.png \
    images/lord_man_1.png \
    images/lord_man_2.png \
    images/lord_win.png \
    images/lord_woman_1.png \
    images/lord_woman_2.png \
    images/lord-right.png \
    images/number.png \
    images/pass.png \
    images/pass_btn-1.png \
    images/pass_btn-2.png \
    images/pass_btn-3.png \
    images/plane_1.png \
    images/plane_2.png \
    images/plane_3.png \
    images/plane_4.png \
    images/plane_5.png \
    images/progress.png \
    images/qiangdizhu.png \
    images/score1.png \
    images/score2.png \
    images/score3.png \
    images/shunzi.png \
    images/start-1.png \
    images/start-2.png \
    images/start-3.png \
    images/logo.ico
