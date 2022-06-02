---
layout: post
title:  "Building a message bank webapp"
categories: blog assignment
permalink: posts/message-bank
author: Pei Xi Kwok
---
Today, we'll be building a message bank webapp using Flask. The webapp will have two pages: one for the user to submit a message, and one for the user to view past messages submitted. The code used to build this webapp will be [available here](https://github.com/kwpexi/message-bank). 

## 1. The basic components

In order to create a functional webapp, there are a few components that we need to get things going. To use Flask, make sure that you have Flask installed and that you enable the development environment through your command line interface using `set FLASK_ENV=development` (for Windows).

You'll want to create a folder titled `app`, containing an `__init__.py` file and two additional folders: one for templates titled `templates`, and another for the CSS style sheet titled `static`. All your html templates will be stored in `templates`, while your `style.css` file will be stored in `static`.

## 2. Setting up subsmissions

We'll start off by creating a `submit` template with three user interface elements:

- A text box for submitting a message.
- A text box for submitting the name of the user.
- A “submit” button.

To do this, we need to create a `submit.html` template and two functions for database management: `get_message_db()` and `insert_message()`. These functions will make use of SQL queries to store the messages submitted.

```python
def get_message_db():
  """
  Checks for a database and a table to store the messages and creates them if they don't already exist

  Returns a connection to the database
  """
  try:
     return g.message_db
  except:
    # opens connection to database
     g.message_db = sqlite3.connect("messages_db.sqlite")

     # SQL query to create messages table within db
     cmd = \
        """
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,handle TEXT,message TEXT)
        """
    
     cursor = g.message_db.cursor()
     cursor.execute(cmd)
     return g.message_db

def insert_message(request):
    """
    Inserts user message into the database

    request - request object from user input
    """
    # extracts message and handle from request object
    msg = request.form["message"]
    name = request.form["name"]

    # SQL query to insert user input into table
    insert_cmd = \
        """
        INSERT INTO messages(handle,message)
        VALUES (?,?)
        """
    cursor = get_message_db().cursor()
    cursor.execute(insert_cmd,(msg,name))
    get_message_db().commit()
    get_message_db().close() 
```

To make use of html template for our submit page, we will put navigation links (the top two links at the top of the screen) inside a template called `base.html`, then have the `submit.html` template extend `base.html`. We will then write a function to `render_template()` the `submit.html` template.

```python
# this goes into __init__.py

@app.route("/", methods=['POST', 'GET'])
def submit():
    if request.method == 'GET': # if user is simply visiting the page
        return render_template('submit.html')
    else:
        try:
            insert_message(request)
            return render_template('submit.html',thanks=True)
        except:
            return render_template('submit.html')
```

```python
# submit.html template
{% raw %}
{% extends 'base.html' %}

{% block header %}
  <h1>{% block title %}Deposit a message{% endblock %}</h1>
{% endblock %}

{% block content %}
    <form method = "post">
        <label for="message">What message do you have?</label>
        <input type="text" name="message" id="message">
        <br>
        <label for="name">Your name</label>
        <input type="text" name="name" id="name">
        <br>
        <br>
        <input type="submit" value="Submit message">
    </form>

{% if thanks %}
    <h1>Thank you for your message deposit.</h1>
{% endif %}
{% endblock %}
{% endraw %}
```
## 3. Viewing random submissions

Next, we'll be able to view random submissions by writing a function called `random_messages(n)` which will return a collection of n random messages from the `message_db`.

```python
def random_messages(n=5):
    """
    Accesses and stores n number of random messages from the messages database

    n - cap on number of random messages

    Returns a collection of n number of random messages from the database
    """
    random_cmd = \
        """
        SELECT *
        FROM messages
        ORDER BY RANDOM()
        LIMIT ?
        """
    cursor = get_message_db().cursor()
    cursor.execute(random_cmd,(str(n)))
    store = cursor.fetchall()
    get_message_db().close()

    return store
```
We will then write a new template called `view.html` to display the messages extracted from `random_messages()`.

```python
# view.html
{% raw %}
{% extends 'base.html' %}

{% block header %}
  <h1>{% block title %}Previous deposits{% endblock %}</h1>
{% endblock %}

{% block content %}
Here are some messages that were previously deposited.
{%for message in messages %}
    <p>"{{message[1]}}"</p>
    <p>- <i> {{message[2]}} </i></p>
    <br>
{% endfor %}
    
{% endblock %}
{% endraw %}
```
Finally, we will write a function to render the `view.html` template.
```python
# this goes in the __init__.py file
@app.route("/view")
def view():
    view_msg = random_messages()
    return render_template('view.html',messages=view_msg)
```
## 4. Changing the style sheet

Last but not least, you can also play around with the elements in the style sheet to change up the appearance of the website. For example, you can try adding a `background-image` for your webapp or changing the font. Here's an example of how your message bank can look!

![hw5_1.png](/images/hw5_1.png)
![hw5_2.png](/images/hw5_2.png)

