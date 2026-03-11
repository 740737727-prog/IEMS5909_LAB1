# Before You Start

## Login to the Jetson

TAs have assigned each group a Jetson Orin NX. You can find the Jetson Orin NX IP address and username/password in the email. If you don't have the email, please contact the TAs. The username in the email is `nvidia` by default. It is a superuser account created for you to access the Jetson Orin NX and create your own user account.

## Connect to the Internet

You need to connect to IE VPN to access the Jetsons. You can find the VPN setup instructions in [IE VPN](IE_PVN.pdf). Special thanks to Limo for providing this detailed guide.

## Connect to the Jetson

You can connect to the Jetson Orin NX using the following command:

```
ssh nvidia@<Jetson Orin NX IP address>
```

Replace `<Jetson Orin NX IP address>` with the actual IP address of the Jetson Orin NX. Enter the password when prompted.

Once you are connected to the Jetson Orin NX, you can start the following steps.

## Create a New User Account

You can create a new user account by running the following command:

```
sudo adduser <username>
```

Replace `<username>` with the username you want to create. You also need to assign sudo and docker permissions to the new user account you just created. Run the following commands using the `nvidia` superuser account to assign the permissions:

```
sudo usermod -aG sudo <username>
sudo usermod -aG docker <username>
```

After the account is created, you can login to the Jetson Orin NX using the new username and password. And you should be able to see `/home/<username>` directory. Place all your code and data in this directory so you won't affect other group members' work.

You can verify the new user account by running the following command to switch to your own user account.

```
sudo su <username>
```

Test sudo and docker permissions:

```
sudo whoami
# should print `root`

docker ps
# should print the list of running containers if any
```

Congratulations! You have successfully set up the prerequisites for the lab. You can continue to the next step and have a try on Edge AI.