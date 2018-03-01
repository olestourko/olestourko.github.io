---
layout: post
title: "Using Marshmallow's Validator classes" 
categories: [python, marshmallow]
image: 'assets/posts/2018-02-28/marshmallow-logo.png'
---

![Marshmallow](/assets/posts/2018-02-28/marshmallow-logo.png){:width="150px"}

I ran into an issue trying to do some validation on a [Marshmallow](http://marshmallow.readthedocs.io/en/latest/quickstart.html#validation)
schema recently. In my case, I wanted a field of type `List` to be required, and also have a min and max length.
 
The documentation doesnt seem to directly explain how to do this, or I didnt look deep enough, but I did find these
[validator classes](https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.validate) in the API
reference.

After some fiddling with it, here's how you do it:

{% highlight python %} 
class SubmitSchema(Schema):
    ...
    perks = fields.List(fields.Integer(), required=True, validate=Length(min=1, max=10))
    ...
    
{% endhighlight %}

`Length` extends `Validator`, which has a `__call__` method that Marshmallow automatically passes a value to when it validates the schema:

{% highlight python %} 
def __call__(self, value):
    length = len(value)

    if self.equal is not None:
        if length != self.equal:
            raise ValidationError(self._format_error(value, self.message_equal))
        return value

    if self.min is not None and length < self.min:
        message = self.message_min if self.max is None else self.message_all
        raise ValidationError(self._format_error(value, message))

    if self.max is not None and length > self.max:
        message = self.message_max if self.min is None else self.message_all
        raise ValidationError(self._format_error(value, message))

    return value
{% endhighlight %}

If you want to use more than on validation, you can do that too:

{% highlight python %} 
class SubmitSchema(Schema):
    ...
    perks = fields.List(fields.Integer(), required=True, validate=[
        Length(min=1, max=10),
        ContainsOnly([1, 2, 3])
    ])
    ...
    
{% endhighlight %}

As it turns out Validator classes aren't the only option, either:

> You can perform additional validation for a field by passing it a validate callable (function, lambda, or object with __call__ defined).

---

### Resources

[Validation Quick Guide](http://marshmallow.readthedocs.io/en/latest/quickstart.html#field-validators-as-methods)  
[API Reference for Fields](https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.fields)  
[API Reference for Validation](https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.validate)