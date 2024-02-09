# Start with the same CentOS Stream 8 as on the course server.
FROM quay.io/centos/centos:stream8

# Configure any "quality of life" improvements.
RUN dnf install -y glibc-langpack-en

# Allow packages to be installed from EPEL.
# https://docs.fedoraproject.org/en-US/epel/
RUN <<EOF
dnf install -y dnf-plugins-core
dnf config-manager --set-enabled powertools
dnf install -y epel-release epel-next-release
EOF

# Install some of the basic utilities used during lecture.
RUN dnf install -y screen htop iotop jq

# Install the most recent Java 8 JDK.
RUN dnf install -y java-1.8.0-openjdk-devel

# Install the most recent release of Apache Hadoop.
ARG HADOOP_VERSION=3.3.6
RUN <<EOF
curl https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz |
  tar -xzf - --directory /opt --owner root --group root --no-same-owner
EOF

# Add the Hadoop tools to our PATH.
COPY <<EOF /etc/profile.d/hadoop.sh
export JAVA_HOME=/usr/lib/jvm/java-1.8.0/jre
export HADOOP_HOME=/opt/hadoop-${HADOOP_VERSION}
export PATH="\${HADOOP_HOME}/bin:\${PATH}"
EOF

# Adjust a few MapReduce configurations to improve performance.
COPY <<EOF /opt/hadoop-${HADOOP_VERSION}/etc/hadoop/mapred-site.xml
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>mapreduce.local.map.tasks.maximum</name>
    <value>4</value>
  </property>
  <property>
    <name>mapreduce.local.reduce.tasks.maximum</name>
    <value>4</value>
  </property>
  <property>
    <name>mapreduce.reduce.shuffle.input.buffer.percent</name>
    <value>0.15</value>
  </property>
</configuration>
EOF

# Provide an environment similar to login as root via SSH.
RUN rm /run/nologin
CMD ["login", "-f", "root", "-p"]
