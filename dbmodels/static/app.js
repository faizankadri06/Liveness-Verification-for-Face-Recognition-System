const form = document.querySelector('form');
form.addEventListener('submit', (event) => {
  event.preventDefault();
  const username = document.querySelector('#username').value;
  const password = document.querySelector('#password').value;
  const role = document.querySelector('#role').value;
  // Check if the user is an admin or a regular user
  // Redirect the user to the appropriate dashboard page
});

var modal = document.getElementById('id01');

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}