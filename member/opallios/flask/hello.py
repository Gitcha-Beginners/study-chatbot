#coding: utf-8

from flask import Flask, request, session, render_template, redirect, url_for, abort
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world!'

@app.route('/main')
def main():
    return 'Main Page'

@app.route('/user/<username>')
def showUserProfile(username):
    app.logger.debug('RETRIEVE DATA - USER ID : %s' % username)
    app.logger.debug('RETRIEVE DATA - Check Complete')
    app.logger.debug('RETRIEVE DATA - Warning... User Not Found.')
    app.logger.debug('RETRIEVE DATA - ERR! User unauthenification.')
    return 'USER : %s' % username

@app.route('/user/id/<int:userId>')
def showUserProfileById(userId):
    return 'USER ID : %d' % userId

@app.route('/account/login', methods=['POST'])
def login():
    if request.method == 'POST':
        userId = request.form['id']
        wp = request.form['wp']

        if len(userId) == 0 or len(wp) == 0:
            return userId + ', ' + wp + ' 로그인 정보를 제대로 입력하지 않았습니다.'.decode('utf-8')

        session['logFlag'] = True
        session['userId'] = userId
        return session['userId'] + ' 님 환영합니다'.decode('utf-8')
    else:
        return '잘못된 접근입니다'.decode('utf-8')

@app.route('/user', methods=['GET'])
def getUser():
    if session.get('logFlag') != True:
        return '잘못된 접근입니다'.decode('utf-8')

    userId = session['userId']

    if 'userId' in session:
        return '[GET][USER] USER ID : {0}'.format(userId)
    else:
        abort(400)

@app.route('/account/logout', methods=['POST', 'GET'])
def logout():
    session['logFlag'] = False
    session.pop('userId', None)
    return redirect(url_for('main'))

@app.errorhandler(400)
def uncaughtError(error):
    return '잘못된 사용입니다.'.decode('utf-8')

@app.errorhandler(404)
def not_found(error):
    resp = make_response(render_template('error.html'), 404)
    resp.headers['X-Something'] = 'A value'
    return resp

@app.route('/login', methods=['POST','GET'])
def login_direct():
    if request.method == 'POST':
        return redirect(url_for('login'), code=307)
    else:
        return redirect(url_for('login'))

app.secret_key = 'sample_secret_key'



if __name__ == '__main__':
    app.debug = True
    app.run(port=8080)