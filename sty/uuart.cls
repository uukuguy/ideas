% !Mode:: "TeX:UTF-8"
%%%%% 'uuart.cls' BEGIN

%%%%%%%%%% class clarification %%%%%%%%%%
% 模板声明

\NeedsTeXFormat{LaTeX2e}[2007/10/19]
\ProvidesClass{uuart}
              [2015/10/01 v0.1 The LaTeX template for exprement report ]
\typeout{Document Class 'uuart' v0.1 by uukuguy (2015/10)}

%%%%%%%%%% class options %%%%%%%%%%
% 模板选项

\newif\ifbuaa@master\buaa@masterfalse
\DeclareOption{master}{\buaa@mastertrue}

\newif\ifbuaa@zhfive\buaa@zhfivefalse
\DeclareOption{zhfive}{\buaa@zhfivetrue}

% 其余选项传递给ctexart
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ProcessOptions\relax
% 引用ctexart及基本设置

\LoadClass[cs5size,a4paper,fancyhdr]{ctexart}[2007/10/19]

%%%%%%%%%% global package %%%%%%%%%%
% 全局通用宏包

\RequirePackage{ifthen}
\RequirePackage{setspace}

\setmainfont{Times New Roman}

%%%%%%%%%% page margin %%%%%%%%%%
% 页面边距

%\usepackage{geometry}
\RequirePackage{geometry}
\newgeometry{
    top=25mm, bottom=25mm, left=20mm, right=20mm,
    headsep=5mm, headheight=10mm, footskip=10mm,
}
\savegeometry{papergeometry}
\loadgeometry{papergeometry}

\renewcommand{\baselinestretch}{1.5}
\setlength{\columnsep}{2em} % 两栏间距
%\setlength{\columnseprule}{1} % 两栏间分割线宽度，默认值为0。
\setlength{\parskip}{0em} % 段落间距
\setlength{\lineskip}{1.5pt} % 行间距
\setlength{\parindent}{2em} % 段落缩进为2个基本字体。
\setlength{\floatsep}{3pt plus 3pt minus 2pt}      % 图形之间或图形与正文之间的距离
\setlength{\abovecaptionskip}{10pt plus 1pt minus 1pt} % 图形中的图与标题之间的距离
\setlength{\belowcaptionskip}{3pt plus 1pt minus 2pt} % 表格中的表与标题之间的距离


% 文章标题
\def\VARtitle{}
\renewcommand{\title}[1]{
    \def\VARtitle{#1}
}

% 文章作者
\def\VARauthor{}
\renewcommand{\author}[1]{
    \def\VARauthor{#1}
}

% 作者单位
\def\VARaddress{}
\newcommand{\address}[1]{
    \def\VARaddress{#1}
}

% 关键词
\def\VARkeywords{}
\newcommand{\keywords}[1]{
    \def\VARkeywords{#1}
}

% 中文摘要
\def\VARcnabstract{}
\newcommand{\cnabstract}[1]{
    \def\VARcnabstract{#1}
}

%%%%%%%%%% header & footer %%%%%%%%%%
% 页眉页脚

%\usepackage{fancyhdr}
\RequirePackage{fancyhdr}
\fancypagestyle{frontmatter}{
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\headrulewidth}{0pt}
    \fancyhf{}
    \fancyfoot[C]{\thepage}

}
\fancypagestyle{mainmatter}{
    \fancyhead{}
    \fancyfoot{}
    \if@twoside
        \fancyhead[CO]{\zihao{-5}\songti
            \VARtitle \vspace{1.5mm}
        }
        \fancyhead[CE]{\zihao{-5}\songti\leftmark\vspace{1.5mm}}
    \else
        \fancyhead[C]{\zihao{-5}\songti
            \ifthenelse{\isodd{\value{page}}}
                { \VARtitle }
                {\leftmark}
            \vspace{1.5mm}
        }
    \fi
    \fancyfoot[C]{\zihao{5}\thepage}

    \renewcommand{\headrulewidth}{0.5bp} % 页眉线宽度
}

\fancypagestyle{plain}{
    \pagestyle{fancy}
}

\renewcommand{\maketitle}{%
    \pagestyle{frontmatter}
    \begin{strip}
    \articletitle
    \end{strip}
}

%%%%%%%%%% title %%%%%%%%%%
% 标题

% 汉化
\renewcommand{\contentsname}{目\qquad 录}
\renewcommand\listfigurename{插\ 图\ 目\ 录}
\renewcommand\listtablename{表\ 格\ 目\ 录}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
% 格式
% 章标题小三号团体居中
%\ctexset { chapter = {
    %format={\zihao{-3}\heiti},
    %nameformat={},
    %aftername={\quad},
    %titleformat={},
    %name={,},
    %number={\chinese{chapter}},
    %beforeskip={-.5\baselineskip},
    %afterskip={\baselineskip},
%}}
% 节标题小四号黑体居左
\ctexset { section = {
    format={\zihao{-4}\heiti},
    aftername={\quad},
    beforeskip={.5\baselineskip},
    afterskip={.5\baselineskip},
}}
% 条标题五号黑体居左
\ctexset { subsection = {
    format={\zihao{5}\heiti},
    aftername={\quad},
    beforeskip={.5\baselineskip},
    afterskip={.5\baselineskip},
}}
% 段标题五号宋体居左
\ctexset { subsubsection = {
    format={\zihao{5}\songti},
    aftername={\quad},
    beforeskip={.5\baselineskip},
    afterskip={.5\baselineskip},
}}

% 网址排版
\usepackage[colorlinks,linkcolor=black,anchorcolor=black,citecolor=black,CJKbookmarks=True]{hyperref}

% 算法排版
\usepackage{algorithm}
\usepackage{algorithmic}
\floatname{algorithm}{算法}
\renewcommand{\algorithmicrequire}{ {输入：}}
\renewcommand{\algorithmicensure}{ {输出：}}
%\renewcommand{\algorithmicreturn}{ {返回}}

% 定义定理环境
%\newtheoremstyle{mystyle}{3pt}{3pt}{\kaishu}{0cm}{\heiti}{\heiti}{}{1em}{}
%\theoremstyle{mystyle}
\newtheorem{definition}{\hspace{2em}定义}
\newtheorem{theorem}{\hspace{2em}定理}
\newtheorem{corollary}{\hspace{2em}推论}
\newtheorem{remark}[definition]{\hspace{2em}注}
\newtheorem{proposition}[definition]{\hspace{2em}命题}
\newtheorem{axiom}[definition]{\hspace{2em}公理}
\newtheorem{lemma}[definition]{\hspace{2em}引理}

% 解决pandoc转换后，tightlist编译出错问题
\newcommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

%%%%%%%%%% abstract %%%%%%%%%%
% 摘要

% 中文摘要
\newenvironment{cabstract}{%
    \vspace*{2bp}
    {\zihao{-5}\heiti 摘要：}
    }{%

        {\zihao{-5}\heiti 关键词：}{\zihao{-5}\songti \VARkeywords}
}

%%%%%%%%%% reference %%%%%%%%%%
% 参考文献
\usepackage[authoryear,square,sort&compress]{natbib}
\bibpunct{[}{]}{,}{n}{}{}
\setlength{\bibsep}{0pt}
\usepackage{hypernat}
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}
\renewcommand{\citet}[1]{\citeauthor{#1} (\citeyear{#1})}
\renewcommand{\citep}[1]{(\citeauthor{#1}, \citeyear{#1})}
\renewcommand\bibname{参\ 考\ 文\ 献} % for report, book
%\renewcommand\refname{参\ 考\ 文\ 献}  % for article
\bibliographystyle{unsrtnat}

% 双栏设置
% 分栏不用另起一页，同一页即可以有双栏也可以有单栏
% 双栏最后一页上左右基本同高
% 如果需要插入单栏内容，只需要把单栏内容放在如下环境中即可：
% \begin{strip}
%   单栏内容
% \end{strip}
% 如果双栏最后一页先排滿左栏，然后再排右栏，需要使用命令
% \raggedend
% 恢复双栏平衡排列的命令为：
% \flushend
\usepackage{flushend,cuted}

% 单、双面设置
% 选用towside格式(book类文档默认选项)，奇数面码打印在页面右边，偶数页码打印在页面左边。每章开始位置默认值openright，书的每个新的一单总是从奇数页开始，可能产生空白的偶数页；openany则表示每章从新的一页开始，不管这一页是奇数页还是偶数页。
% 选用oneside格式(article类文档和report类文档默认选项)，所有页码的打印位置相同。


%%%%%%%%%% table %%%%%%%%%%
% 表格

% 长表格
\usepackage{longtable}

% 表格中的行合并
\usepackage{multirow}

% 重定义table/tabular/tabularx环境，使表格内为5号字
% TODO(huxuan): 支持 longtable
\let\oldtable\table
\let\endoldtable\endtable
\renewenvironment{table}[1][h!]
{\renewcommand{\arraystretch}{1.5}
\oldtable[#1]\zihao{5}}
{\renewcommand{\arraystretch}{1}
\endoldtable}

\let\oldtabular\tabular
\let\endoldtabular\endtabular
\renewenvironment{tabular}[1][h!]
{\renewcommand{\arraystretch}{1.5}
\oldtabular[#1]\zihao{5}}
{\renewcommand{\arraystretch}{1}
\endoldtabular}

\usepackage{tabularx}
\let\oldtabularx\tabularx
\let\endoldtabularx\endtabularx
\renewenvironment{tabularx}[2]
{\renewcommand{\arraystretch}{1.5}
\zihao{5}\oldtabularx{#1}{#2}}
{\renewcommand{\arraystretch}{1}
\endoldtabularx}


\usepackage{array}


%\footnote[0]{\scriptsize{\receivedate, \modifydate.
%\\
%\mbox{\;}\hspace{2\ccwd}\doino		
 %\\ \authorsinfo}}
%\vspace{-1em}
%

%%%%%%%%%%%% Article Title %%%%%%%%%%%%%%

\newcommand{\articletitle}{
    \begin{spacing}{1.2}
    \begin{center}
        \begin{minipage}[c]{16cm}
        %\vskip 0bp
        \centerline{\zihao{3}\ziju{0.0}\heiti \VARtitle }
        \vskip 20bp
        \centerline{\zihao{5}\ziju{0.0}\songti \VARauthor }
        \vskip 5bp
        \centerline{\zihao{-5}\ziju{0.0}\songti \VARaddress }
        \vskip 10bp
        \begin{cabstract}
            {\zihao{-5}\songti \VARcnabstract}
        \end{cabstract}
        \end{minipage}
    \end{center}
    \end{spacing}
    %\vskip 10bp
}

