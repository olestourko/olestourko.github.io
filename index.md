---
title: Oles Tourko
---

## About Me

Software Developer, Maker, Lagomorph herder

![No Context](/assets/index/rabbit.jpg)

Email: [otourko@mailbox.org](mailto:otourko@mailbox.org)  
LinkedIn: [olestourko](https://www.linkedin.com/in/olestourko/)  
CV / Resumé: [download PDF]({{ 'assets/download/oles-tourko-cv.pdf' | absolute_url }})

---

## Recent Posts
<div>
{% for post in site.posts limit:10 %}
  <a href="{{ post.url }}">{{ post.title }} - {{post.date | date: "%-d %B %Y"}}</a><br>
{% endfor %}
</div>