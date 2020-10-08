# ML-Musketeers' blog

ML-Musketeers' data science blog.

Hosted on github pages.

Built following [this blog post](https://www.dataquest.io/blog/how-to-setup-a-data-science-blog/).

## Setup

## Automated publishing with MAKE

(from the repo root folder)

- To test locally: `make local` (it will open chrome and run the pelican server)
- To publish changes: `make github`.

## Edit the template file

### Template

Mostly you'll edit the `base.html` file in `themes/neat/static/templates`.

### Style

1. Install `less`: `npm install -g less`.
2. Install the following plugin: `npm install -g less-plugin-clean-css`.
3. Go to the `themes/neat/static/stylesheet` folder
4. Edit the `style.less` and `variables.less` files
5. run `lessc -clean-css style.less style.min.css` to generate the `style.min.css` file

## TODO

- [ ] Add Goodle Analytics support
- [ ] Improve the notebook template
