function togglePassword(checkbox) {
  const targetId = checkbox.getAttribute('data-target');
  const field = document.getElementById(targetId);
  if (field) {
    field.type = field.type === "password" ? "text" : "password";
  }
}
class Users {
    constructor(username, first_name, last_name, email, password) {
        this.username = username;
        this.first_name = first_name;
        this.last_name = last_name;
        this.email = email;
        this.password = password;
    }
}

function rregistered(event) {
  event.preventDefault();
  let user = {
    username: document.getElementById("username").value,
    first_name: document.getElementById("first_name").value,
    last_name: document.getElementById("last_name").value,
    email: document.getElementById("email").value,
    password: document.getElementById("password").value,
    confirm_password: document.getElementById("confirm_password").value,
  }; 
  if (Object.values(user).some(value => value.trim() === "")) {
      alert("Please fill in all fields.");
      return false;
  } else if (user.password !== user.confirm_password) {
      alert("Passwords do not match!");
      return false;
  } else {
      // Create a new user object for debugging purposes
      const newUser = new Users(user.username, user.first_name, user.last_name, user.email, user.password);
      console.log(newUser);

      // TODO: Implement logic to send the user data to a backend server
      alert("Registration successful!");
      window.location.href = "main.html";
  }
}

