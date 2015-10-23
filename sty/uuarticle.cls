% !Mode:: "TeX:UTF-8"
%%%%% 'expreport.cls' BEGIN

%%%%%%%%%% class clarification %%%%%%%%%%
% 模板声明

\NeedsTeXFormat{LaTeX2e}[2007/10/19]
\ProvidesClass{expreport}
              [2015/10/01 v0.1
The LaTeX template for exprement report ]
\typeout{Document Class `exprement' v0.1 by uukuguy (2015/10)}


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

%%%%%%%%%% font %%%%%%%%%%
% 数学相关和字体设置

\RequirePackage{amsmath,amssymb,amsfonts,mathrsfs,bm}
%\RequirePackage[amsmath,thmmarks,hyperref]{ntheorem}
\RequirePackage[amsmath,thmmarks]{ntheorem}
\RequirePackage{txfonts}

% 主要字体为Times New Roman和宋体
% 模板某些标题需要华文行楷和32号字
\setmainfont{Times New Roman}
% 不需要设置CJKmainfont，ctex 宏包已经很好的处理了
% 不仅设置了粗体为黑体，斜体为楷体，还兼容了winfonts和adobefonts
% 直接设置反而会在只有adobefonts的情况下报错
% \setCJKmainfont{宋体}
\setCJKfamilyfont{hwxingkai}{STXingkai}
\newcommand{\hwxingkai}{\CJKfamily{hwxingkai}}
\newcommand{\xiaochuhao}{\fontsize{32pt}{\baselineskip}\selectfont}

%%%%%%%%%% color %%%%%%%%%%
% 颜色设置

% 只用于电子版
\RequirePackage{color}
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

%%%%%%%%%% page margin %%%%%%%%%%
% 页面边距

\RequirePackage{geometry}
%\newgeometry{
    %top=30mm, bottom=25mm, left=30mm, right=20mm,
    %headsep=5mm,
%}
%\savegeometry{bachelorgeometry}
\newgeometry{
    top=25mm, bottom=25mm, left=20mm, right=20mm,
    headsep=5mm, headheight=10mm, footskip=10mm,
}
\savegeometry{reportgeometry}

\loadgeometry{reportgeometry}

%%%%%%%%%% space %%%%%%%%%%
% 其他间距

\renewcommand{\baselinestretch}{1.5}
\setlength{\columnsep}{2em} % 两栏间距
%\setlength{\columnseprule}{1} % 两栏间分割线宽度，默认值为0。
\setlength{\parskip}{0em} % 段落间距
\setlength{\lineskip}{1.5pt} % 行间距
\setlength{\parindent}{2em} % 段落缩进为2个基本字体。
\setlength{\floatsep}{3pt plus 3pt minus 2pt}      % 图形之间或图形与正文之间的距离
\setlength{\abovecaptionskip}{10pt plus 1pt minus 1pt} % 图形中的图与标题之间的距离
\setlength{\belowcaptionskip}{3pt plus 1pt minus 2pt} % 表格中的表与标题之间的距离

%%%%%%%%%% header & footer %%%%%%%%%%
% 页眉页脚

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
            \expreport@subject 实验报告\vspace{1.5mm}
        }
        \fancyhead[CE]{\zihao{-5}\songti\leftmark\vspace{1.5mm}}
    \else
        \fancyhead[C]{\zihao{-5}\songti
            \ifthenelse{\isodd{\value{page}}}
                {\expreport@subject 实验报告}
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


%%%%%%%%%% title %%%%%%%%%%
% 标题

% 汉化
\renewcommand{\contentsname}{目\qquad 录}
\renewcommand\listfigurename{插\ 图\ 目\ 录}
\renewcommand\listtablename{表\ 格\ 目\ 录}
\newcommand\bibname{参\ 考\ 文\ 献}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
% 格式
% 章标题小三号团体居中
%\ctexset { chapter = {
    %format={\centering\zihao{-3}\heiti},
    %%format={\zihao{3}\heiti},
    %nameformat={},
    %aftername={\quad},
    %titleformat={},
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

\ifbuaa@master
    \ctexset { chapter = {
        format={\zihao{-3}\heiti},
        %name={第,章xxxx},
        name={,},
        number={\chinese{chapter}},
    }}
\fi

%%%%%%%%%% contents %%%%%%%%%%
% 目录

\RequirePackage{titletoc}
% 一级目录
\titlecontents{chapter}[0pt]{\vspace{-0.25\baselineskip}\heiti\zihao{4}}{\thecontentslabel\ }{}
    {\hspace{.5em}\titlerule*[4pt]{$\cdot$}{\zihao{-4}\contentspage}}[\vspace{0.07\baselineskip}]
% 二级目录
\titlecontents{section}[2em]{\heiti\zihao{-4}}{\thecontentslabel\ }{}
{\hspace{.5em}\titlerule*[4pt]{$\cdot$}{\zihao{-4}\contentspage}}[\vspace{0.1\baselineskip}]
% 三级目录
\titlecontents{subsection}[3.5em]{\vspace{-0.2\baselineskip}\songti\zihao{5}}{\thecontentslabel\ }{}
    {\hspace{.5em}\titlerule*[4pt]{$\cdot$}{\zihao{-4}\contentspage}}[\vspace{0.1\baselineskip}]
% 四级目录
\titlecontents{subsubsection}[4.5em]{\vspace{-0.2\baselineskip}\heiti\zihao{-5}}{\thecontentslabel\ }{}
    {\hspace{.5em}\titlerule*[4pt]{$\cdot$}{\zihao{-4}\contentspage}}[\vspace{0.1\baselineskip}]

% 取消图片、表格目录中的章节空格
\newcommand*{\noaddvspace}{\renewcommand*{\addvspace}[1]{}}
\addtocontents{lof}{\protect\noaddvspace}
\addtocontents{lot}{\protect\noaddvspace}

%表目录图目录的格式设置
%表目录与图目录数字前增加“表”与“图”字，并且使目录行间距与section行间距一致

\titlecontents{figure}[0pt]{\vspace{0.15\baselineskip}\songti\zihao{-4}}{\makebox[3em][l]{图~\thecontentslabel\quad}}{}
    {\hspace{.5em}\titlerule*[4pt]{$\cdot$}\contentspage}[\vspace{0.15\baselineskip}]

\titlecontents{table}[0pt]{\vspace{0.15\baselineskip}\songti\zihao{-4}}{\makebox[3em][l]{表~\thecontentslabel\quad}}{}
    {\hspace{.5em}\titlerule*[4pt]{$\cdot$}\contentspage}[\vspace{0.15\baselineskip}]


%%%%%%%%%% reference %%%%%%%%%%
% 参考文献

%\bibliographystyle{expreport} % 参考文献格式
\RequirePackage[authoryear,square,sort&compress]{natbib}
\bibpunct{[}{]}{,}{n}{}{}
\setlength{\bibsep}{0pt}
\RequirePackage{hypernat}
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}
\bibliographystyle{unsrtnat}
%\bibliographystyle{unsrt}
%\bibliographystyle{IEEEtran}
%\bibliographystyle{expreport_orig}

\renewcommand{\citet}[1]{\citeauthor{#1} (\citeyear{#1})}
\renewcommand{\citep}[1]{(\citeauthor{#1}, \citeyear{#1})}

%%%%%%%%%% table %%%%%%%%%%
% 表格

% 长表格
\RequirePackage{longtable}

% 表格中的行合并
\RequirePackage{multirow}

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

\RequirePackage{tabularx}
\let\oldtabularx\tabularx
\let\endoldtabularx\endtabularx
\renewenvironment{tabularx}[2]
{\renewcommand{\arraystretch}{1.5}
\zihao{5}\oldtabularx{#1}{#2}}
{\renewcommand{\arraystretch}{1}
\endoldtabularx}


\RequirePackage{array}

%%%%%%%%%% picture %%%%%%%%%%
% 图片

\RequirePackage{graphicx}
\DeclareGraphicsExtensions{.eps,.ps,.jpg,.bmp}
\graphicspath{{figure/}}
\RequirePackage{pifont} % “秘级”后的五角星
\RequirePackage{subfigure}

%%%%%%%%%% list %%%%%%%%%%
% 列表

\RequirePackage{enumitem}
\setlist{noitemsep}
\setlist[1,2]{labelindent=\parindent}
\setlist[enumerate,1]{label=\arabic*、}
\setlist[enumerate,2]{label=（\arabic*）}
\setlist{
    topsep=0pt,
    itemsep=0pt,
    partopsep=0pt,
    parsep=\parskip,
}

%%%%%%%%%% code %%%%%%%%%%
% 代码

% Listing 的设置请参考 http://en.wikibooks.org/wiki/LaTeX/Packages/Listings
\RequirePackage{listings}
\lstset{
    backgroundcolor=\color{white},
    basicstyle=\zihao{5}\ttfamily,
    columns=flexible,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    frame=single,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    stepnumber=1,
    rulecolor=\color{black},
    tabsize=2,
    texcl=true,
    title=\lstname,
    escapeinside={\%*}{*)},
    extendedchars=false,
    mathescape=true,
    xleftmargin=3em,
    xrightmargin=3em,
}
%\ifbuaa@color
    %\lstset{
        %numberstyle=\color{gray},
        %keywordstyle=\color{blue},
        %commentstyle=\color{dkgreen},
        %stringstyle=\color{mauve},
    %}
%\else
    \lstset{
        numberstyle=\color{black},
        keywordstyle=\color{black},
        commentstyle=\color{black},
        stringstyle=\color{black},
    }
%\fi

% 重命名Listings标题头
\renewcommand{\lstlistingname}{代码}

%%%%%%%%%% file directory %%%%%%%%%%
% 文件目录

\RequirePackage{dirtree}
%\ifbuaa@color
    %\renewcommand*\DTstylecomment{\ttfamily\color{dkgreen}}
    %\renewcommand*\DTstyle{\ttfamily\color{red}}
%\fi

%%%%%%%%%% caption %%%%%%%%%%
% 图表标题

\RequirePackage{caption}
\DeclareCaptionFormat{masterfigure}{\bf\songti\zihao{5}{#1\textrm{#2}#3}}
\DeclareCaptionFormat{mastertable}{\bf\songti\zihao{5}{#1\textrm{#2}#3}}
\DeclareCaptionFormat{masterlstlisting}{\bf\songti\zihao{5}{#1\textrm{#2}#3}}
\RequirePackage{remreset}
\@removefromreset{table}{chapter}
\@removefromreset{figure}{chapter}
%使图表的标号与章节无关
\captionsetup[figure]{format=masterfigure,labelsep=quad}
\captionsetup[table]{format=mastertable,labelsep=quad}
\captionsetup[lstlisting]{format=masterlstlisting,labelsep=quad}
\renewcommand{\thefigure}{\arabic{figure}}
\renewcommand{\thetable}{\arabic{table}}
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}

% Fix subfigure reference bug
\let\p@subfigure=\thefigure

% 图片表格标题命令，主要用于混排
\newcommand\figcaption{\def\@captype{figure}\caption}
\newcommand\tabcaption{\def\@captype{table}\caption}

%%%%%%%%%% other settings %%%%%%%%%%
% 杂项

% 设置<附录>的图表编号与当前附录章号关联
\newcommand{\rmrelation}{
% 图、表、公式编号随 chapter 清零
\@addtoreset{figure}{chapter}
\@addtoreset{table}{chapter}
\@addtoreset{equation}{chapter}
%图、表、公式编号章节关联
\renewcommand{\thefigure}{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\renewcommand{\thetable}{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\renewcommand{\theequation}{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
}

\let\oldappendix\appendix
\renewcommand{\appendix}{
\oldappendix\rmrelation
}


% 下划线
\RequirePackage{ulem}

% 设置行距
\RequirePackage{setspace}

% 正文前的页码设置位大写罗马数字
\renewcommand{\frontmatter}{
    \cleardoublepage
    \@mainmatterfalse
    \ifbuaa@bachelor
        \pagenumbering{Roman}
    \else
        \pagenumbering{roman}
    \fi
}

% 保证偶数页结束章节
\newcommand{\clearemptydoublepage}{%
    \clearpage
    \if@twoside
        \ifodd
            \c@page
        \else
            \hbox{}\thispagestyle{empty}\newpage
            \if@twocolumn
                \hbox{}\newpage
            \fi
        \fi
    \fi
}


%% 可配置选项
\setcounter{tocdepth}{3} % 显示4级目录
\setcounter{secnumdepth}{3} % 4级标题编号

% 报告别名
\def\expreport@reportnickname{}
\newcommand{\reportnickname}[1]{
    \def\expreport@reportnickname{#1}
}

% 实验名称
\def\expreport@subject{}
\newcommand{\subject}[1]{
    \def\expreport@subject{#1}
}

%%%%%%%%%% common user info %%%%%%%%%%
% 用户信息

% 实验名称
%\def\expreport@subject{负面舆情判定}

% 解决pandoc转换后，tightlist编译出错问题
\newcommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

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
\newtheorem{definition}{\hspace{2em}定义}[chapter]
\newtheorem{theorem}{\hspace{2em}定理}[chapter]
\newtheorem{corollary}{\hspace{2em}推论}[chapter]
\newtheorem{remark}[definition]{\hspace{2em}注}
\newtheorem{proposition}[definition]{\hspace{2em}命题}
\newtheorem{axiom}[definition]{\hspace{2em}公理}
%\newtheorem{lemma}[definition]{\hspace{2em}引理}

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


