---
title: Oles Tourko
---

## About Me

Software Developer, Maker, Rabbit herder

[CV / Resumé]({{ 'assets/download/oles-tourko-cv.pdf' | absolute_url }})

Email: [otourko@mailbox.org](mailto:otourko@mailbox.org)  
LinkedIn: [olestourko](https://www.linkedin.com/in/olestourko/)

---

## Recent Posts
<div>
{% for post in site.posts limit:10 %}
  <a href="{{ post.url }}">{{ post.title }} - {{post.date | date: "%-d %B %Y"}}</a><br>
{% endfor %}
</div>