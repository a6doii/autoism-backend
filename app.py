import os
from Autism import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 11000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
