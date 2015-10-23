all: TermGraph

TermGraph_1:
	pandoc TermGraph.md -t latex | sed 's/\\_/_/g' | sed 's/\\section/\\chapter/g' |  sed 's/\\subsection/\\section/g' |  sed 's/\\subsubsection/\\subsection/g' |  sed 's/\\paragraph/\\subsubsection/g'  > TermGraph.tex
	xelatex report_sample.tex
	-bibtex  report_sample.aux
	xelatex report_sample.tex

clean:
	find . -name '*.aux' -print0 | xargs -0 rm -rf
	rm -rf *.lof *.log *.lot *.out *.toc *.bbl *.blg *.thm
	rm -f context/*.tex

depclean: clean
	rm -rf *.pdf

context/%.tex: context/%.md
	pandoc context/$*.md -t latex | sed 's/\\_/_/g' | sed 's/\\section/\\chapter/g' |  sed 's/\\subsection/\\section/g' |  sed 's/\\subsubsection/\\subsection/g' |  sed 's/\\paragraph/\\subsubsection/g'  > context/$*.tex

report_%.pdf: context/%.tex report_%.tex 
	xelatex report_$*
	bibtex  report_$*.aux
	xelatex report_$*
	xelatex report_$*

TermGraph: report_TermGraph.pdf 

