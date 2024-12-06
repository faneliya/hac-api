from flask import Flask, request, jsonify
from flask import API

route = API(__name__)

@route.route('/')
def test_code():
    return { "data": "success"}

