---
title: Oles Tourko
---

## About Me

I am a software developer, mostly using Python and PHP.
My current job is full-stack web developer at [Digital Extremes](http://www.digitalextremes.com/) in London, Ontario.
My background is in Computer Science and I've been doing web development professionally for the past 4 years. Backend and full-stack roles are my favourite when it comes to web.

I'm also a machine learning hobbyist and am very interested in how we can use it to solve problems. Investing and sustainability are areas I like, and my personal projects are _usually_ related to them.

[My CV / Resumé]({{ 'assets/download/oles-tourko-cv.pdf' | absolute_url }})

## Contact

Email: [olestourko@gmail.com](mailto:olestourko@gmail.com)  
LinkedIn: [olestourko](https://www.linkedin.com/in/olestourko/)

---

## Recent Posts
<div>
{% for post in site.posts limit:10 %}
  <a href="{{ post.url }}">{{ post.title }} - {{post.date | date: "%-d %B %Y"}}</a><br>
{% endfor %}
</div>