# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tinyopt'
copyright = '2025, Julien Michot'
author = 'Julien Michot'

# Check version
def ReadVersion(filepath = "../../cmake/Version.cmake"):
    import re
    major, minor, patch = [None, None, None]
    try:
        with open(filepath, 'r') as f:
            for line in f:
                major_match = re.search(r'set\(TINYOPT_VERSION_MAJOR\s+(\d+)\)', line)
                minor_match = re.search(r'set\(TINYOPT_VERSION_MINOR\s+(\d+)\)', line)
                patch_match = re.search(r'set\(TINYOPT_VERSION_PATCH\s+(\d+)\)', line)
                if major_match:
                    major = major_match.group(1)
                if minor_match:
                    minor = minor_match.group(1)
                if patch_match:
                    patch = patch_match.group(1)

                if major is not None and minor is not None and patch is not None:
                    break  # Found all parts, no need to continue

        if major is not None and minor is not None and patch is not None:
            return f"{major}.{minor}.{patch}"
        else:
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None



release = ReadVersion()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    #'sphinx_sitemap',
    'sphinx.ext.inheritance_diagram',
    'myst_parser',
    'breathe'
]


templates_path = ['_templates']
exclude_patterns = []

highlight_language = 'c++'

root_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',  #  Provided by Google in your dashboard
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,

    'logo_only': False,

    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# html_logo = ''
# github_url = ''
# html_baseurl = ''

html_static_path = ['_static']

breathe_projects = {
	'tinyopt_docs': "../../build/xml/"
}
breathe_default_project = 'tinyopt_docs'
breathe_default_members = ('members', 'undoc-members')