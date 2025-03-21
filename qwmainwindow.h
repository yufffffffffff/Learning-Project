#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_qwmainwindow.h"
#include "qformdoc.h"

class QWMainWindow : public QMainWindow
{
    Q_OBJECT

public:
	QWMainWindow(QWidget *parent = nullptr);
    ~QWMainWindow();

private:
    Ui::qwmainwindowClass ui;

private:
	// �����ڹر�ʱ�ر������Ӵ���
	void closeEvent(QCloseEvent *event);

private slots:
	void on_actDoc_New_triggered();   

	void on_actDoc_Open_triggered();

	void on_actCut_triggered();

	void on_actFont_triggered();

	void on_actCopy_triggered();

	void on_actPaste_triggered();	

	void on_mdiArea_subWindowActivated(QMdiSubWindow *argl);  // �Ӵ��ڱ�����

	void on_actViewMode_triggered(bool checked);  // MDIģʽ����

	void on_actCascade_triggered();

	void on_actTile_triggered();

	void on_actCloseALL_triggered();


};
