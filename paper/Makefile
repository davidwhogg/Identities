# WARNING: WILL FAIL ON STUFF OLDER THAN 2011 BECAUSE OF pdflatex ISSUES

RM = /bin/rm -f

.SUFFIXES: .tex .dvi .ps .pdf

all: id.pdf

.tex.pdf:
	pdflatex $*
	pdflatex $*
	pdflatex $*

clean:
	$(RM) *.log *.aux *.dvi *.toc *.bbl *.blg

spotless: clean
	$(RM) *.pdf *.ps
