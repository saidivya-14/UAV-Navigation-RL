# 🚀 UAV Navigation using Deep Reinforcement Learning

## 📌 Project Overview
This project implements **autonomous UAV navigation** using **Deep Q-Learning (DQN)**. The UAV learns to navigate a grid-based environment while avoiding obstacles and efficiently reaching a target position.

## 🎯 Features
- 🧠 **Deep Q-Network (DQN)** for reinforcement learning.
- 🚧 **Obstacle avoidance** using dataset-based obstacle mapping.
- 🎯 **Reward-based training** to optimize UAV movements.
- 📈 **Visualization of training performance** using Matplotlib.
- ⚡ **Efficient training** with experience replay and epsilon decay.

## 📂 Project Structure
```
UAV_Navigation_Project/
│-- uav_navigation.py  # Main RL training script
│-- UAV_full_data.csv  # Dataset for obstacle mapping
│-- README.md          # Project documentation
│-- requirements.txt   # Dependencies

```

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/UAV-Navigation-RL.git
   cd UAV-Navigation-RL
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Project
To train the UAV agent, run the following command:
```bash
python uav_navigation.py
```

## 📊 Training Performance Visualization
After training, a **reward plot** is generated showing how the UAV's performance improves across episodes.

### Example Output:
```
✅ Episode 10: Total Reward = -20500
✅ Episode 20: Total Reward = -14500
✅ Episode 30: Total Reward = -9000
✅ Episode 40: Total Reward = -3500
✅ Episode 50: Total Reward = 1000 (UAV successfully reaches the target!)
```

### Example Reward Plot:
![Training Progress][(https://via.placeholder.com/800x400?text=Training+Reward+Plot](https://github.com/saidivya-14/UAV-Navigation-RL/blob/main/UAV_output.jpg))  
*(Replace the placeholder with the actual generated plot)*

## 🔮 Future Enhancements
- ✅ Add **dynamic obstacles** for real-time UAV navigation challenges.
- ✅ Extend navigation to **3D space** with altitude control.
- ✅ Integrate with UAV simulation frameworks like **AirSim** or **Gazebo**.
- ✅ Implement **multi-agent UAV coordination** strategies.

## 📄 Requirements
The project requires the following Python libraries:
- numpy
- tensorflow
- keras
- matplotlib
- pandas

Install them using:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing
We welcome contributions! Feel free to fork the repository, make changes, and submit a pull request. For major changes, open an issue first to discuss what you'd like to improve.

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

### 🔗 Connect with Me
💼 **LinkedIn:** [Your Profile](www.linkedin.com/in/saidivya-kodipaka)  
📂 **GitHub:** [Your Repos]([https://github.com/your-username](https://github.com/saidivya-14))  
📧 **Email:** saidivyakodipaka856@gmail.com

