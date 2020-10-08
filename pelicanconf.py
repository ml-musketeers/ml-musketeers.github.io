#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'The Machine Learning Musketeers'
SITENAME = "ML-Musketeers"
SITEURL = 'https://ml-musketeers.github.io'
SITETITLE = "ML-Musketeers"
SITESUBTITLE = "A gentlemen's journey in Reinforcement Learning"
SITEDESCRIPTION = "A gentlemen's journey in Reinforcement Learning"
SITELOGO = SITEURL + '/images/PhotoThiebaut.jpg'
FAVICON = SITEURL + '/images/favicon.ico'

PATH = 'content'
MAIN_MENU = True
STATIC_PATHS = ['images', 'pdfs']

TIMEZONE = 'US/Pacific'
THEME = os.path.join(os.getcwd(), "themes", "neat")

DEFAULT_LANG = 'en'

SOCIAL = (('linkedin', 'https://www.linkedin.com/in/nthiebaut'),
          ('github', 'https://github.com/nkthiebaut/'),
          ('twitter', 'https://twitter.com/NicoThiebaut'),)

MENUITEMS = (('Archives', '/archives.html'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
MARKUP = ('md', 'ipynb')

from pelican_jupyter import markup as nb_markup
PLUGINS = [nb_markup]

PLUGIN_PATHS = ['./plugins']
PLUGINS = ['share_post.share_post']
IGNORE_FILES = ['.ipynb_checkpoints']
