FROM alpine:latest

# copy crontabs for root user
COPY config/cronjobs /etc/crontabs/root
COPY .ssh /root/.ssh
COPY backup.sh /root/backup.sh
RUN chmod +x /root/backup.sh
RUN apk add --no-cache btrfs-progs coreutils libressl
RUN apk add --no-cache bash openssh-client pv bzip2 curl xz btrbk
# start crond with log level 8 in foreground, output to stderr
CMD ["crond", "-f", "-d", "8"]

