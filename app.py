from Autism import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11000, debug=False, use_reloader=False)
