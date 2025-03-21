#include "qwmainwindow.h"
#include <QMdiSubWindow>
#include <QDir>
#include <QFileDialog>

QWMainWindow::QWMainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	this->setCentralWidget(ui.mdiArea);
	// 窗口最大化显示
	this->setWindowState(Qt::WindowMaximized);
	ui.mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

QWMainWindow::~QWMainWindow()
{}

void QWMainWindow::closeEvent(QCloseEvent *event)
{
	// 关闭所有子窗口
	ui.mdiArea->closeAllSubWindows();
	event->accept();
}

// 新建一个QFormDoc文档
void QWMainWindow::on_actDoc_New_triggered()
{
	QFormDoc *formDoc = new QFormDoc(this);
	// 文档窗口添加到MDI
	ui.mdiArea->addSubWindow(formDoc);
	formDoc->show();
}

// 打开文件
void QWMainWindow::on_actDoc_Open_triggered()
{
	// 是否需要新建子窗口
	bool needNew = false;  
	QFormDoc *formDoc;
	// 如果已有打开的主窗口，需要获取活动窗口
	if (ui.mdiArea->subWindowList().count() > 0)
	{
		formDoc = (QFormDoc*)ui.mdiArea->activeSubWindow()->widget();
		needNew = formDoc->isFileOpened();   // 文件已经打开，需要新建窗口
	}
	else
		needNew = true;

	// 获取系统当前目录
	QString curPath = QDir::currentPath();
	// 对话框标题
	QString dlgTitle = QString::fromLocal8Bit("打开一个文件");
	// 文件过滤器
	QString filter = QString::fromLocal8Bit("文本文件(*.txt);;图片文件(*.jpg *.gif *.png);;所有文件(*.*)");

	QString fileName = QFileDialog::getOpenFileName(this, dlgTitle, curPath, filter);

	if (fileName.isEmpty())
		return;

	// 需要新建子窗口
	if (needNew)
	{
		// 指定父窗口，必须在父窗口为Widget窗口提供一个显示区域
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

// 设置字体
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

// 当前活动子窗口切换时
void QWMainWindow::on_mdiArea_subWindowActivated(QMdiSubWindow *argl)
{
	Q_UNUSED(argl);
	// 若子窗口个数为零
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
		ui.statusBar->showMessage(formDoc->currentFileName());   // 显示主窗口的文件名

	}
}

// MDI显示模式
void QWMainWindow::on_actViewMode_triggered(bool checked)
{
	// tab多页显示模式
	if (checked)
	{
		// tab多页显示模式
		ui.mdiArea->setViewMode(QMdiArea::TabbedView);
		ui.mdiArea->setTabsClosable(true);   // 页面可关闭
		ui.actCascade->setEnabled(false);
		ui.actTile->setEnabled(false);
	}
	else
	{
		// 子窗口模式
		ui.mdiArea->setViewMode(QMdiArea::SubWindowView);
		ui.actCascade->setEnabled(true);
		ui.actTile->setEnabled(true);
	}
}

// 窗口级联展开
void QWMainWindow::on_actCascade_triggered()
{
	ui.mdiArea->cascadeSubWindows();
}

void QWMainWindow::on_actTile_triggered()
{//平铺展开
	ui.mdiArea->tileSubWindows();
}

void QWMainWindow::on_actCloseALL_triggered()
{//关闭全部子窗口
	ui.mdiArea->closeAllSubWindows();
}