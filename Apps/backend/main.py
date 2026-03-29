from flask_socketio import SocketIO,emit
from flask import Flask,request,jsonify
from Apps.Neural_Net.model_for_web import NeuralNet
from Apps.Neural_Net.RELU import RELU
from Apps.Neural_Net.optim import SGD
from Apps.Neural_Net.optim import ADAM,RMSprop,Adagrad,Momentum
from Apps.Neural_Net.loss import MSE
from flask_cors import CORS
import numpy as np
app=Flask(__name__)
CORS(app, origins=["http://localhost:5173"])
socketio=SocketIO(app,cors_allowed_origins="*")

model=NeuralNet(activation=RELU)
X = np.linspace(-1, 1, 100).reshape(-1,1)
y = X**3 + 0.1*np.random.randn(100,1)

def compute_loss_surface(model, X, y):
    w1_vals = np.linspace(-2, 2, 50)
    w2_vals = np.linspace(-2, 2, 50)
    
    loss_surface = np.zeros((50, 50))
    
    # Save original weights
    orig_w1 = model.fc1.weight.copy()
    orig_w2 = model.fc2.weight.copy()
    
    for i, w1 in enumerate(w1_vals):
        for j, w2 in enumerate(w2_vals):
            
            model.fc1.weight[0,0] = w1
            model.fc2.weight[0,0] = w2
            
            y_hat = model.forward(X)
            loss = MSE(y_hat, y)
            
            loss_surface[i,j] = loss.item()
    
    # Restore weights
    model.fc1.weight = orig_w1
    model.fc2.weight = orig_w2
    
    return w1_vals, w2_vals, loss_surface
def train(data):
    optimizer_name = data['optimizer'].upper()  # normalize casing
    
    if optimizer_name == 'SGD':
        optimizer = SGD(params=[model.fc1, model.fc2], lr=data['lr'])
    elif optimizer_name == 'ADAM':
        optimizer = ADAM(params=[model.fc1, model.fc2], lr=data['lr'])
    elif optimizer_name == 'MOMENTUM':
        optimizer = Momentum(params=[model.fc1, model.fc2], lr=data['lr'])  # replace with real one
    elif optimizer_name == 'RMSPROP':
        optimizer = RMSprop(params=[model.fc1, model.fc2], lr=data['lr'])  # replace with real one
    elif optimizer_name == 'ADAGRAD':
        optimizer = Adagrad(params=[model.fc1, model.fc2], lr=data['lr'])  # replace with real one
    else:
        print(f"Unknown optimizer: {data['optimizer']}")
        return  # exit cleanly
    
    for epoch in range(data['epochs']):
        out=model.forward(X)
        loss=MSE(out,y)
        dout=loss.backward()
        model.backward(dout)
        optimizer.step()
        optimizer.zero_grad()
        w1=float(model.fc1.weight[0,0])
        w2=float(model.fc2.weight[0,0])
        if(epoch%10==0):
            socketio.emit("update",{
                "w1":w1,
                "w2":w2,
                "loss":float(loss.item())
            })
            socketio.sleep(0)
            print(f"Epoch {epoch} emitted: w1={w1:.3f}, w2={w2:.3f}")
@app.route('/start',methods=['POST'])
def start_updates():
    data=request.json
    w1_vals,w2_vals,loss_surface=compute_loss_surface(model,X,y)
    socketio.emit("surface",{
        "w1":w1_vals.tolist(),
        "w2":w2_vals.tolist(),
        "loss":loss_surface.tolist()
    })
    socketio.start_background_task(train, data)
    return jsonify({"status": "started"})
    


if __name__ == "__main__":
    socketio.run(app, debug=True)




