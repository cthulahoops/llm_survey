I need to take this json:

{
  'first_name': 'Jane',
  'last_name': 'Smith',
  'date_of_birth': '03/16/1977',
  orders: [
    {
      'created': '2024-01-25T15:00:04',
      'amount': '100.00'
    }
]}

And generate some dataclasses.

date_of_birth needs to be a date.
created is a datetime.
amount is a decimal.

How would I do this using the marshmallow library in python?
