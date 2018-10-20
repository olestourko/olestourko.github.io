---
title: Oles Tourko
---

## About Me

I'm a software developer with professional experience in digital commerce, video games, and most recently insurance.
Right now Python is my main language for work and play. I'm also a value investor for fun, and sometimes even profit!

[My CV / Resumé]({{ 'assets/download/oles-tourko-cv.pdf' | absolute_url }})

## Contact

Email: [olestourko@gmail.com](mailto:olestourko@gmail.com)  
LinkedIn: [olestourko](https://www.linkedin.com/in/olestourko/)

Feel free to add me on Telegram if we know each other from the real world somehow.

---

## Recent Posts
<div>
{% for post in site.posts limit:10 %}
  <a href="{{ post.url }}">{{ post.title }} - {{post.date | date: "%-d %B %Y"}}</a><br>
{% endfor %}
</div>