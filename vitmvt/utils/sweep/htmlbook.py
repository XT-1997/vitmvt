import base64
import os
from io import BytesIO

import matplotlib.pyplot as plt
from yattag import Doc, indent


class Htmlbook:
    """An Htmlbook is used to generate an html page from text and matplotlib
    figures."""

    def __init__(self, args, title):
        """Initializes Htmlbook with a given title."""
        # The doc is used for the body of the document
        self.doc, self.tag, self.text, self.line = Doc().ttl()
        # The top_doc is used for the title and table of contents
        self.top_doc, self.top_tag, self.top_text, self.top_line = Doc().ttl()
        # Add link anchor and title to the top_doc
        self.top_line('a', '', name='top')
        self.top_line('h1', title)
        self.section_counter = 1
        self.sweep_out_dir = args.sweep_out_dir
        self.html_file = os.path.join(self.sweep_out_dir, 'analysis.html')

    def add_section(self, name):
        """Adds a section to the Htmlbook (also updates table of contents)."""
        anchor = 'section{:03d}'.format(self.section_counter)
        name = str(self.section_counter) + ' ' + name
        anchor_style = 'text-decoration: none;'
        self.section_counter += 1
        # Add section to main text
        self.doc.stag('br')
        self.doc.stag('hr', style='border: 2px solid')
        with self.tag('h3'):
            self.line('a', '', name=anchor)
            self.text(name + ' ')
            self.line('a', '[top]', href='#top', style=anchor_style)
        # Add section to table of contents
        self.top_line('a', name, href='#' + anchor, style=anchor_style)
        self.top_doc.stag('br')

    def add_plot(self, matplotlib_figure, ext='svg', file_name=None, **kwargs):
        """Adds a matplotlib figure embedded directly into the html."""
        out = BytesIO()
        matplotlib_figure.savefig(
            out, format=ext, bbox_inches='tight', **kwargs)
        if file_name:
            self.fig = os.path.join(self.sweep_out_dir,
                                    '{}.png'.format(file_name))
            plt.savefig(self.fig)
        plt.close(matplotlib_figure)
        if ext == 'svg':
            self.doc.asis('<svg' + out.getvalue().decode().split('<svg')[1])
        else:
            out = base64.b64encode(out.getbuffer()).decode('ascii')
            self.doc.asis("<img src='data:image/{};base64,{}'/>".format(
                ext, out))
        self.doc.stag('br')

    def add_details(self, summary, details):
        """Adds a collapsible details section to Htmlbook."""
        with self.tag('details'):
            self.line('summary', summary)
            self.line('pre', details)

    def to_text(self):
        """Generates a string representing the Htmlbook (including figures)."""
        return indent(self.top_doc.getvalue() + self.doc.getvalue())

    def to_file(self, out_file=None):
        """Saves Htmlbook to a file (typically should have .html extension)."""
        if out_file is not None:
            with open(out_file, 'w') as file:
                file.write(self.to_text())
        else:
            with open(self.html_file, 'w') as file:
                file.write(self.to_text())
