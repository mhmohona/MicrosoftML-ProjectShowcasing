## Author: Aleem Juma

from flask import render_template, request
from app import app
from app.quotes import get_random_word, find_matching_quote

@app.route('/', methods=['GET','POST'])
def index():
    # generate 4 random words
    genres = [get_random_word() for _ in range(4)]
    
    # if no requested word, pick the first random word
    genre = request.args.get('genre', default=genres[0])
    app.logger.info(f'Request received: "{genre}"')
    
    # find matching quote
    quote, author, matches = find_matching_quote(genre, 4)
    
    # display the page
    return render_template(
        'quote_finder.html',
        requested=genre,        # requested word (or first random word)
        genres=genres,          # selected random words
        quote=quote,            # selected quote
        author=author,          # author of selected quote
        matches=matches         # best matches
    )
