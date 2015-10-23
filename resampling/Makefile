all: report_resampling.pdf

clean:
	find . -name '*.aux' -print0 | xargs -0 rm -rf
	rm -rf *.lof *.log *.lot *.out *.toc *.bbl *.blg *.thm
	rm -fr output
	rm -f context/*.tex

depclean: clean
	rm -rf *.pdf

context/%.tex: context/%.md
	pandoc context/$*.md -t latex | sed 's/\\_/_/g' | sed 's/\\section/\\chapter/g' |  sed 's/\\subsection/\\section/g' |  sed 's/\\subsubsection/\\subsection/g' |  sed 's/\\paragraph/\\subsubsection/g'  > context/$*.tex

report_%.pdf: context/%.tex main.tex 
	mkdir -p output
	xelatex -output-directory=./output main
	bibtex  ./output/main.aux
	xelatex -output-directory=./output main
	xelatex -output-directory=./output main
	cp output/main.pdf report_resampling.pdf


