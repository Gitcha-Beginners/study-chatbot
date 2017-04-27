# _*_ coding: utf-8 _*_

from flask import Flask, request, session, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world!'


@app.route('/main')  # declare URL patern and POST method.
def main():
    return 'Main Page'

@app.route('/user/<username>')  # `<>` can handle the URL pattern
def showUserProfile(username):
    app.logger.debug('RETRIEVE DATA - USER ID : %s' % username)
    app.logger.debug('RETRIEVE DATA - Check Complete')
    app.logger.warn('RETRIEVE DATA - Warning.. User not found')
    app.logger.error('RETRIEVE DATA - ERR! user unauthentifacted')
    return 'USER : %s' % username

@app.route('/user/id/<int:userId>')  # `datatype: ` -> only this datatype can be entered.
def showUserProfileById(userId):
    return 'USER ID : %d' % userId

# Log-in function and creating a session.
@app.route('/account/login', methods=['POST']) # REST Action Type
def login():
    if request.method == 'POST':
        userId = request.form['id']
        wp = request.form['wp']

        if len(userId) == 0 or len(wp) == 0:
            return userId + ', '+ wp + 'you missed the user id or password. Please type again.'

        session['logFlag'] = True
        session['userId'] = userId
        return session['userId'] + ', welcome to this page.'
    else:
        return 'Fails to access.'

app.secret_key = 'simple_secret_key'  # if the session is created, this is mandatory.
                                      # if not, Flask will return the 500 error.

# Access to the login information
@app.route('/user', methods=['GET'])
def getUser():
    if session.get('logFlag') != True:
        return 'Fails to access'
    userId = session['userId']
    return '[GET][USER] USER ID : {0}'.format(userId)

# Logout
@app.route('/account/logout', methods=['POST', 'GET'])
def logout():
    session['logFlag'] = False
    session.pop('userId', None)
    return redirect(url_for('main'))  # url_for(): call the declared function.

@app.errorhandler(400)
def uncaughtError(error):
    return 'worng usage'

@app.errorhandler(404)
def not_found(error):
    resp = make_response(render_template('error.html'), 404)
    resp.headers['X-Something'] = 'A value'
    return resp

# Abort example
# @app.route('/user', methods=['GET'])
# def getUser():
#     if 'userId' in session:
#         return '[GET][USER] USER ID : {0}'.format(session['userId'])
#     else:
#         abort(400)

# redirect
@app.route('/login', methods=['POST', 'GET'])
def login_direct():
    if request.method == 'POST':
        return redirect(url_for('login'), code=307)
    else:
        return redirect(url_for('login'))




if __name__ == '__main__':
    app.debug = True  # if this is True, the server is restarted if there is some changes in the source code.
    app.run()
