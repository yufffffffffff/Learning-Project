#include "qwmainwindow.h"
#include <QMdiSubWindow>
#include <QDir>
#include <QFileDialog>

QWMainWindow::QWMainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	this->setCentralWidget(ui.mdiArea);
	// ���������ʾ
	this->setWindowState(Qt::WindowMaximized);
	ui.mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

QWMainWindow::~QWMainWindow()
{}

void QWMainWindow::closeEvent(QCloseEvent *event)
{
	// �ر������Ӵ���
	ui.mdiArea->closeAllSubWindows();
	event->accept();
}

// �½�һ��QFormDoc�ĵ�
void QWMainWindow::on_actDoc_New_triggered()
{
	QFormDoc *formDoc = new QFormDoc(this);
	// �ĵ�������ӵ�MDI
	ui.mdiArea->addSubWindow(formDoc);
	formDoc->show();
}

// ���ļ�
void QWMainWindow::on_actDoc_Open_triggered()
{
	// �Ƿ���Ҫ�½��Ӵ���
	bool needNew = false;  
	QFormDoc *formDoc;
	// ������д򿪵������ڣ���Ҫ��ȡ�����
	if (ui.mdiArea->subWindowList().count() > 0)
	{
		formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
		needNew = formDoc->isFileOpened();   // �ļ��Ѿ��򿪣���Ҫ�½�����
	}
	else
		needNew = true;

	// ��ȡϵͳ��ǰĿ¼
	QString curPath = QDir::currentPath();
	// �Ի������
	QString dlgTitle = QString::fromLocal8Bit("��һ���ļ�");
	// �ļ�������
	QString filter = QString::fromLocal8Bit("�ı��ļ�(*.txt);;ͼƬ�ļ�(*.jpg *.gif *.png);;�����ļ�(*.*)");

	QString fileName = QFileDialog::getOpenFileName(this, dlgTitle, curPath, filter);

	if (fileName.isEmpty())
		return;

	// ��Ҫ�½��Ӵ���
	if (needNew)
	{
		// ָ�������ڣ������ڸ�����ΪWidget�����ṩһ����ʾ����
		formDoc = new QFormDoc(this);
		ui.mdiArea->addSubWindow(formDoc);
	}

	formDoc->loadFromFile(fileName);
	formDoc->show();

	ui.actCut->setEnabled(true);
	ui.actCopy->setEnabled(true);
	ui.actPaste->setEnabled(true);
	ui.actFont->setEnabled(true);

}

void QWMainWindow::on_actCut_triggered()
{
	QFormDoc* formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
	formDoc->textCut();
}

// ��������
void QWMainWindow::on_actFont_triggered()
{
	QFormDoc* formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
	formDoc->textCopy();
}

void QWMainWindow::on_actCopy_triggered()
{
	QFormDoc* formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
	formDoc->textCopy();
}

void QWMainWindow::on_actPaste_triggered()
{
	QFormDoc* formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
	formDoc->textPaste();
}

// ��ǰ��Ӵ����л�ʱ
void QWMainWindow::on_mdiArea_subWindowActivated(QMdiSubWindow *argl)
{
	Q_UNUSED(argl);
	// ���Ӵ��ڸ���Ϊ��
	if (ui.mdiArea->subWindowList().count() == 0)
	{
		ui.actCut->setEnabled(false);
		ui.actCopy->setEnabled(false);
		ui.actPaste->setEnabled(false);
		ui.actFont->setEnabled(false);
		ui.statusBar->clearMessage();
	}
	else
	{
		QFormDoc *formDoc = static_cast<QFormDoc*>(ui.mdiArea->activeSubWindow()->widget());
		ui.statusBar->showMessage(formDoc->currentFileName());   // ��ʾ�����ڵ��ļ���

	}
}

// MDI��ʾģʽ
void QWMainWindow::on_actViewMode_triggered(bool checked)
{
	// tab��ҳ��ʾģʽ
	if (checked)
	{
		// tab��ҳ��ʾģʽ
		ui.mdiArea->setViewMode(QMdiArea::TabbedView);
		ui.mdiArea->setTabsClosable(true);   // ҳ��ɹر�
		ui.actCascade->setEnabled(false);
		ui.actTile->setEnabled(false);
	}
	else
	{
		// �Ӵ���ģʽ
		ui.mdiArea->setViewMode(QMdiArea::SubWindowView);
		ui.actCascade->setEnabled(true);
		ui.actTile->setEnabled(true);
	}
}

// ���ڼ���չ��
void QWMainWindow::on_actCascade_triggered()
{
	ui.mdiArea->cascadeSubWindows();
}

void QWMainWindow::on_actTile_triggered()
{//ƽ��չ��
	ui.mdiArea->tileSubWindows();
}

void QWMainWindow::on_actCloseALL_triggered()
{//�ر�ȫ���Ӵ���
	ui.mdiArea->closeAllSubWindows();
}