---
layout: page
title: Blog Archive
permalink: /archive/
---

{% if site.tags and site.tags.size > 0 %}
{% assign sorted_tags = site.tags | sort %}
{% for tag in sorted_tags %}
<h3>{{ tag[0] }}</h3>
<ul>
  {% for post in tag[1] %}
    <li>
      <a href="{{ post.url | relative_url }}">
        {{ post.date | date: "%B %Y" }} — {{ post.title }}
      </a>
    </li>
  {% endfor %}
</ul>
{% endfor %}

{% assign untagged = site.posts | where_exp: "p", "p.tags == empty" %}
{% if untagged.size > 0 %}
<h3>Untagged</h3>
<ul>
{% for post in untagged %}
<li>
<a href="{{ post.url | relative_url }}">
{{ post.date | date: "%B %Y" }} — {{ post.title }}
</a>
</li>
  {% endfor %}
</ul>
{% endif %}
{% else %}
<ul>
{% for post in site.posts %}
<li>
<a href="{{ post.url | relative_url }}">
{{ post.date | date: "%B %Y" }} — {{ post.title }}
</a>
</li>
{% endfor %}
</ul>
{% endif %}