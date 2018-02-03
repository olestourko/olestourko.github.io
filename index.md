I'm a [software developer](https://www.linkedin.com/in/olestourko/) and mostly work in Python and PHP. My current job is full stack web developer at
[Digital Extremes](http://www.digitalextremes.com/) in London, Ontario.

---

## Recent posts
{% for post in site.posts limit:10 %}
  [{{ post.title }}]({{ post.url }}) - {{post.date | date: "%-d %B %Y"}}
{% endfor %}
