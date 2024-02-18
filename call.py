# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = "AC57690a6535a5d6cf979d9562e9e13f39"
auth_token = "29414eacdc2e2aacf1633fb4b12b6586"
client = Client(account_sid, auth_token)

call = client.calls.create(
    url="http://demo.twilio.com/docs/voice.xml",
    to="+919359974046",
    from_="+12055765757",
)

print(call.sid)
